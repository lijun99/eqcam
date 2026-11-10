"""Tests without any API: a synthetic video with shaking and a change after it, the schema, the request of each
backend (a fake Anthropic client, a local stand-in for an OpenAI-compatible server), and review and report with
a fake backend"""

import csv
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from eqcam import backends, frames, report, review, schema

FPS, H, W = 30, 360, 640


def example_findings(severity=2):
    fd = {key: {'present': 'no', 'description': '', 'where': ''} for key in schema.DETECTIONS}
    fd.update(camera_status='ok', severity=severity, confidence='medium', summary='Smoke after the shaking.',
              shaking={'visible': 'yes', 'effects': ['poles swaying'], 'intensity_mmi': 'V-VI',
                       'basis': 'poles and wires swaying'})
    fd['smoke_or_dust'] = {'present': 'yes', 'description': 'a plume', 'where': 'upper left, AFTER'}
    return fd


def write_video(path, seconds=12, shake=(4.0, 8.0), change_at=8.0, seed=0, noise=3):
    """A textured scene, shaken by a few pixels between shake[0] and shake[1] s, with a dark patch after
    change_at s"""
    rng = np.random.default_rng(seed)
    scene = ndimage.gaussian_filter(rng.uniform(0, 255, (H + 40, W + 40)), 3)
    scene = (255 * (scene - scene.min()) / np.ptp(scene)).astype(np.uint8)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(int(seconds * FPS)):
        t = k / FPS
        dy = dx = 0
        if shake[0] <= t < shake[1]:
            dy, dx = int(round(6 * np.sin(2 * np.pi * 3 * t))), int(round(8 * np.cos(2 * np.pi * 2.5 * t)))
        img = scene[20 + dy:20 + dy + H, 20 + dx:20 + dx + W].astype(np.int16)
        if t >= change_at:
            img[100:260, 200:440] = 20
        if noise:
            img += rng.integers(-noise, noise + 1, img.shape, dtype=np.int16)    # sensor noise
        proc.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('event')
    write_video(str(root / 'cam1.mp4'))
    write_video(str(root / 'cam2.mp4'), shake=(100, 100), change_at=100, seed=1)    # nothing happens
    with open(root / 'cameras.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['camera_id', 'name', 'lat', 'lon', 'video', 'start_utc', 'view'])
        w.writerow(['cam1', 'Shaken', 34.10, -118.20, 'cam1.mp4', '2026-01-01T00:00:00Z', 'a street'])
        w.writerow(['cam2', 'Quiet', 34.30, -118.40, 'cam2.mp4', '', 'a hillside'])
        w.writerow(['cam3', 'No video', 34.20, -118.30, 'missing.mp4', '', ''])
    with open(root / 'event.json', 'w') as f:
        json.dump(dict(event_id='test', origin_utc='2026-01-01T00:00:00Z', lat=34.0, lon=-118.0, depth_km=8,
                       magnitude=6.5), f)
    out = str(root / 'out')
    frames.prepare(str(root / 'cameras.csv'), str(root), out, str(root / 'event.json'))
    return out


def metrics_of(out):
    with open(os.path.join(out, 'metrics.json')) as f:
        return {m['camera_id']: m for m in json.load(f)}


def test_prepare_finds_the_shaking_and_the_change(prepared):
    m = metrics_of(prepared)
    cam1 = m['cam1']
    assert cam1['shaking_detected']
    assert abs(cam1['onset_s'] - 4.0) < 0.5 and abs(cam1['end_s'] - 8.0) < 0.7
    assert 3 < cam1['shake_amplitude_px'] < 15
    assert 'scene_changed' in cam1['flags'] and cam1['priority'] == 'high'
    assert 'predicted_p_s' in cam1                            # start_utc and the event are known
    with Image.open(os.path.join(prepared, cam1['tile'])) as im:
        assert im.size == (1280, 720)
    assert set(cam1['frames']) == {'before', 'peak', 'after', 'latest'}


def test_shaking_of_a_noiseless_static_scene(tmp_path):
    """Identical calm frames followed by motion are a static scene, not a frozen upload"""
    write_video(str(tmp_path / 'still.mp4'), seconds=8, shake=(3.0, 6.0), change_at=100, noise=0)
    frames_small = frames.decode_small(str(tmp_path / 'still.mp4'), 320, 180, 15)
    sh = frames.find_shaking(frames.frame_differences(frames_small), 15, 0, len(frames_small))
    assert sh['detected'] and abs(sh['onset'] / 15 - 3.0) < 0.5


def test_prepare_quiet_and_missing(prepared):
    m = metrics_of(prepared)
    assert not m['cam2']['shaking_detected'] and m['cam2']['priority'] == 'low'
    assert 'scene_changed' not in m['cam2']['flags']
    assert m['cam3']['flags'] == ['no_video'] and 'tile' not in m['cam3']
    with open(os.path.join(prepared, 'tasks.json')) as f:
        tasks = json.load(f)
    assert tasks[0]['priority'] == 'high'


def test_schema():
    assert schema.validate(example_findings()) == []
    bad = example_findings()
    bad['fire']['present'] = 'maybe'
    del bad['summary']
    bad['extra'] = 1
    problems = schema.validate(bad)
    assert len(problems) == 3
    assert 'Motion measured' in schema.user_text(dict(camera_id='c', name='n', motion_text='none'))


def test_parse_findings():
    fd = example_findings()
    wrapped = '<think>hmm</think>\n```json\n' + json.dumps(fd) + '\n```'
    assert backends.parse_findings(wrapped) == fd
    assert backends.finish('m', 'not json', {}).error.startswith('not JSON')


class FakeMessages:
    def __init__(self, text, stop='end_turn'):
        self.text, self.stop, self.calls = text, stop, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop, model=kwargs['model'], stop_details=None,
                               usage=SimpleNamespace(input_tokens=1500, output_tokens=300),
                               content=[SimpleNamespace(type='thinking', thinking=''),
                                        SimpleNamespace(type='text', text=self.text)])


def fake_anthropic(text, stop='end_turn'):
    messages = FakeMessages(text, stop)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages)), messages


def test_claude_backend():
    client, messages = fake_anthropic(json.dumps(example_findings()))
    b = backends.ClaudeBackend(client=client)
    r = b.review(b'\xff\xd8jpeg', 'Camera: c')
    assert r.error is None and r.findings['severity'] == 2 and r.usage['input_tokens'] == 1500
    call = messages.calls[0]
    assert call['model'] == 'claude-opus-5-5' and call['fallbacks'] == 'default'
    assert call['betas'] == ['server-side-fallback-2026-07-01']
    assert call['output_config']['format']['schema'] is schema.FINDINGS_SCHEMA
    image = call['messages'][0]['content'][0]
    assert image['type'] == 'image' and image['source']['media_type'] == 'image/jpeg'
    # no server-side fallbacks on Vertex AI, nor for Haiku
    for kw in (dict(platform='vertex'), dict(model='claude-haiku-5-5')):
        assert 'fallbacks' not in backends.ClaudeBackend(client=client, **kw).request(b'x', 't')
    refused, _ = fake_anthropic('', stop='refusal')
    assert backends.ClaudeBackend(client=refused).review(b'x', 't').error.startswith('refusal')


def test_gemini_schema():
    def keys(s):
        return {k for k in s} | set().union(*(keys(v) for v in s.values() if isinstance(v, dict))) \
            if isinstance(s, dict) else set()
    stripped = backends.gemini_schema(schema.FINDINGS_SCHEMA)
    assert 'additionalProperties' in keys(schema.FINDINGS_SCHEMA)
    assert 'additionalProperties' not in keys(stripped)


class StandIn(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat completions server that answers with the example findings"""
    received = []

    def do_POST(self):                                    # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        StandIn.received.append(body)
        out = {'model': body['model'], 'usage': {'prompt_tokens': 900, 'completion_tokens': 250},
               'choices': [{'message': {'content': json.dumps(example_findings(3))}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_qwen_backend():
    server = HTTPServer(('127.0.0.1', 0), StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        b = backends.QwenBackend(base_url=f'http://127.0.0.1:{server.server_port}/v1')
        r = b.review(b'\xff\xd8jpeg', 'Camera: c')
    finally:
        server.shutdown()
    assert r.error is None and r.findings['severity'] == 3 and r.usage['input_tokens'] == 900
    body = StandIn.received[-1]
    assert body['response_format']['json_schema']['schema'] is not None
    assert body['messages'][1]['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,')


class FakeBackend:
    name, model = 'fake', 'fake-1'

    def __init__(self, severity):
        self.severity = severity

    def review(self, image, text):
        assert image[:2] == b'\xff\xd8'                     # the tile, a JPEG
        return backends.Result(example_findings(self.severity), self.model, {'input_tokens': 10, 'output_tokens': 5})


def test_review_escalate_and_report(prepared):
    first = review.review(prepared, FakeBackend(2), stage='screen', workers=2)
    assert {r['camera_id'] for r in first} == {'cam1', 'cam2'}     # cam3 has no video
    assert review.review(prepared, FakeBackend(2), stage='screen') == []      # already reviewed
    second = review.review(prepared, FakeBackend(3), stage='final', escalate_from='screen', min_severity=3)
    assert second == []                                             # nothing reached severity 3
    second = review.review(prepared, FakeBackend(3), stage='final', escalate_from='screen', min_severity=2)
    assert len(second) == 2
    assert review.validate_stage(prepared, 'final') == {}
    rows = report.report(prepared)
    assert {r['camera_id']: r['severity'] for r in rows} == {'cam1': 3, 'cam2': 3, 'cam3': None}
    for name in ('survey.csv', 'survey.json', 'map.html', 'map.png'):
        assert os.path.getsize(os.path.join(prepared, name)) > 0
    with open(os.path.join(prepared, 'survey.csv')) as f:
        table = list(csv.DictReader(f))
    assert table[0]['smoke_or_dust'] == 'yes'
