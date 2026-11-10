"""Vision-language backends: each reviews one image (the 2 x 2 tile of a camera) with its text, and returns the
findings as a dict following eqcam.schema.FINDINGS_SCHEMA

  claude   Claude through the Anthropic API, or through Google Vertex AI (platform="vertex")
  gemini   Gemini through Google Vertex AI
  qwen     Qwen3-VL (or another open model) behind an OpenAI-compatible server, e.g., vLLM

The SDKs are imported only by the backend that needs them (pip install "eqcam[claude]", "eqcam[gemini]").
"""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .schema import FINDINGS_SCHEMA, SYSTEM_PROMPT, validate


@dataclass
class Result:
    findings: dict | None
    model: str
    usage: dict = field(default_factory=dict)
    error: str | None = None
    raw: str = ''


def parse_findings(text):
    """The findings from a model's text: the JSON object in it (a model without structured output may wrap it
    in a code fence or add a reasoning block)"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def finish(model, text, usage):
    """A Result from a model's text, with the schema problems as the error"""
    try:
        findings = parse_findings(text)
    except json.JSONDecodeError as err:
        return Result(None, model, usage, f'not JSON: {err}', text)
    problems = validate(findings)
    return Result(findings, model, usage, '; '.join(problems) if problems else None, text)


def with_retries(call, tries=4, wait=5.0, retryable=lambda err: False):
    """Call, retrying with exponential backoff on the errors retryable says are transient"""
    for k in range(tries):
        try:
            return call()
        except Exception as err:                       # noqa: BLE001 - classified by retryable
            if k == tries - 1 or not retryable(err):
                raise
            time.sleep(wait * 2 ** k)


# --------------------------------------------------------------------------------------------- Claude

class ClaudeBackend:
    """Claude, with structured output (the findings schema) and adaptive thinking at the given effort; on the
    Anthropic API, refusals of the models that support it fall back server-side to Anthropic's default model"""

    name = 'claude'
    FALLBACK_MODELS = {'claude-opus-5-5', 'claude-opus-5', 'claude-fable-5-1', 'claude-sonnet-5-5'}

    def __init__(self, model='claude-opus-5-5', platform='anthropic', effort='medium', project=None,
                 region='global', max_tokens=16000, max_retries=5, client=None):
        self.model, self.platform, self.effort, self.max_tokens = model, platform, effort, max_tokens
        if client is None:
            import anthropic
            if platform == 'vertex':
                client = anthropic.AnthropicVertex(project_id=project or os.environ.get('GOOGLE_CLOUD_PROJECT'),
                                                   region=region, max_retries=max_retries)
            elif platform == 'anthropic':
                client = anthropic.Anthropic(max_retries=max_retries)     # credentials from the environment
            else:
                raise ValueError(f'unknown platform {platform!r}: anthropic or vertex')
        self.client = client

    def request(self, image, text):
        """The arguments of the Messages API call"""
        kwargs = dict(
            model=self.model, max_tokens=self.max_tokens,
            system=[{'type': 'text', 'text': SYSTEM_PROMPT, 'cache_control': {'type': 'ephemeral'}}],
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                             'data': base64.standard_b64encode(image).decode()}},
                {'type': 'text', 'text': text}]}],
            output_config={'effort': self.effort,
                           'format': {'type': 'json_schema', 'schema': FINDINGS_SCHEMA}},
        )
        if self.platform == 'anthropic' and self.model in self.FALLBACK_MODELS:
            kwargs.update(betas=['server-side-fallback-2026-07-01'], fallbacks='default')
        return kwargs

    @staticmethod
    def describe_error(err):
        """A short description of an API error (the SDK has already retried the transient ones), or None if
        err is not one"""
        import anthropic
        if isinstance(err, anthropic.BadRequestError):         # the request itself is wrong
            return f'bad request: {err.message}'
        if isinstance(err, anthropic.RateLimitError):          # still limited after the SDK's retries
            return f'rate limited: {err.message}'
        if isinstance(err, anthropic.APIStatusError):
            return f'API error {err.status_code}: {err.message}'
        if isinstance(err, anthropic.APIConnectionError):
            return f'connection error: {err}'
        return None

    def review(self, image, text):
        kwargs = self.request(image, text)
        try:
            if 'betas' in kwargs:
                response = self.client.beta.messages.create(**kwargs)
            else:
                response = self.client.messages.create(**kwargs)
        except Exception as err:                              # noqa: BLE001 - API errors recorded, others raised
            message = self.describe_error(err)
            if message is None:
                raise
            return Result(None, self.model, error=message)
        usage = dict(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        model = getattr(response, 'model', self.model)
        if response.stop_reason == 'refusal':
            details = getattr(response, 'stop_details', None)
            return Result(None, model, usage, f'refusal: {getattr(details, "category", None)}')
        if response.stop_reason == 'max_tokens':
            return Result(None, model, usage, 'stopped at max_tokens')
        text = next((b.text for b in response.content if b.type == 'text'), '')
        return finish(model, text, usage)


# --------------------------------------------------------------------------------------------- Gemini

def gemini_schema(schema):
    """The schema without the keywords Gemini's OpenAPI-style response_schema does not take"""
    if isinstance(schema, dict):
        return {k: gemini_schema(v) for k, v in schema.items() if k != 'additionalProperties'}
    if isinstance(schema, list):
        return [gemini_schema(v) for v in schema]
    return schema


class GeminiBackend:
    """Gemini on Google Vertex AI (google-genai SDK), with JSON output constrained by the findings schema;
    credentials from Application Default Credentials (gcloud auth application-default login)"""

    name = 'gemini'

    def __init__(self, model='gemini-2.5-flash', project=None, location='global', client=None, tries=4):
        self.model, self.tries = model, tries
        if client is None:
            from google import genai
            client = genai.Client(vertexai=True, project=project or os.environ.get('GOOGLE_CLOUD_PROJECT'),
                                  location=location)
        self.client = client

    def config(self):
        from google.genai import types
        common = dict(system_instruction=SYSTEM_PROMPT, response_mime_type='application/json')
        try:                                                  # full JSON Schema, in recent SDKs
            return types.GenerateContentConfig(**common, response_json_schema=FINDINGS_SCHEMA)
        except (TypeError, ValueError):
            return types.GenerateContentConfig(**common, response_schema=gemini_schema(FINDINGS_SCHEMA))

    def review(self, image, text):
        from google.genai import types

        def call():
            return self.client.models.generate_content(
                model=self.model, contents=[types.Part.from_bytes(data=image, mime_type='image/jpeg'), text],
                config=self.config())

        def retryable(err):
            return getattr(err, 'code', None) in (408, 429, 500, 502, 503, 504)
        try:
            response = with_retries(call, self.tries, retryable=retryable)
        except Exception as err:                              # noqa: BLE001 - recorded with the camera
            return Result(None, self.model, error=f'{type(err).__name__}: {err}')
        meta = getattr(response, 'usage_metadata', None)
        usage = dict(input_tokens=getattr(meta, 'prompt_token_count', None),
                     output_tokens=getattr(meta, 'candidates_token_count', None),
                     thinking_tokens=getattr(meta, 'thoughts_token_count', None))
        return finish(self.model, response.text or '', usage)


# ----------------------------------------------------------------------------------------------- Qwen

class QwenBackend:
    """An open vision-language model (default Qwen3-VL) behind an OpenAI-compatible chat completions server,
    such as vLLM (vllm serve Qwen/Qwen3-VL-8B-Instruct); the JSON is constrained by the server's guided
    decoding (response_format json_schema)"""

    name = 'qwen'

    def __init__(self, model='Qwen/Qwen3-VL-8B-Instruct', base_url=None, api_key=None, max_tokens=2048,
                 timeout=300.0, tries=4):
        self.model, self.max_tokens, self.timeout, self.tries = model, max_tokens, timeout, tries
        self.url = (base_url or os.environ.get('EQCAM_QWEN_URL', 'http://localhost:8000/v1')).rstrip('/')
        self.api_key = api_key or os.environ.get('EQCAM_QWEN_API_KEY')

    def request(self, image, text):
        """The body of the chat completions call"""
        data_url = 'data:image/jpeg;base64,' + base64.standard_b64encode(image).decode()
        return {
            'model': self.model, 'max_tokens': self.max_tokens, 'temperature': 0,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_url}},
                                                      {'type': 'text', 'text': text}]}],
            'response_format': {'type': 'json_schema',
                                'json_schema': {'name': 'findings', 'schema': FINDINGS_SCHEMA, 'strict': True}},
        }

    def review(self, image, text):
        body = json.dumps(self.request(image, text)).encode()
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        def call():
            req = urllib.request.Request(f'{self.url}/chat/completions', data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)

        def retryable(err):
            return (isinstance(err, urllib.error.HTTPError) and err.code in (429, 500, 502, 503, 504)) or \
                isinstance(err, (urllib.error.URLError, TimeoutError)) and not isinstance(err, urllib.error.HTTPError)
        try:
            out = with_retries(call, self.tries, retryable=retryable)
        except Exception as err:                              # noqa: BLE001 - recorded with the camera
            return Result(None, self.model, error=f'{type(err).__name__}: {err}')
        usage = out.get('usage') or {}
        usage = dict(input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'))
        text = out['choices'][0]['message'].get('content') or ''
        return finish(out.get('model', self.model), text, usage)


BACKENDS = {'claude': ClaudeBackend, 'gemini': GeminiBackend, 'qwen': QwenBackend}


def make_backend(name, **options):
    """A backend by name, with the options it takes (others are dropped)"""
    cls = BACKENDS[name]
    import inspect
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in options.items() if k in accepted and v is not None})
