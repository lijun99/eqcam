"""The findings of the review of one camera: the JSON schema shared by every backend, the instructions,
and a validator (for findings written by hand or by a model without structured output)"""

YNU = {'type': 'string', 'enum': ['yes', 'no', 'uncertain']}


def _detection(what):
    return {'type': 'object', 'description': what, 'additionalProperties': False,
            'required': ['present', 'description', 'where'],
            'properties': {'present': YNU,
                           'description': {'type': 'string', 'description': 'what is seen; empty if no'},
                           'where': {'type': 'string', 'description': 'where in the image, e.g., "left of the '
                                     'bridge, AFTER frame"; empty if no'}}}


# the effects looked for, in the order of the table and the map
DETECTIONS = {
    'structural_damage': 'collapse or partial collapse, fallen facades or parapets, broken windows, tilted buildings',
    'infrastructure_damage': 'damaged bridges, roads, rails, pipes; downed poles or wires; water main breaks',
    'landslide': 'landslides, rockfall, slope failures, new scars on hillsides',
    'ground_failure': 'ground cracks, liquefaction (sand boils, settlement, tilted objects), surface rupture',
    'fire': 'flames or glow',
    'smoke_or_dust': 'smoke plumes, dust clouds',
    'power_outage': 'lights going out, signals dark',
    'water': 'flooding, sloshing pools or tanks, seiche, unusual waves',
    'people_response': 'people running, evacuating or gathering (counts and actions only, never who they are)',
}

FINDINGS_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['camera_status', 'shaking', *DETECTIONS, 'severity', 'confidence', 'summary'],
    'properties': {
        'camera_status': {'type': 'string', 'enum': ['ok', 'dark', 'frozen', 'moved', 'obstructed', 'unusable'],
                          'description': 'the state of the camera and its view after the shaking'},
        'shaking': {
            'type': 'object', 'additionalProperties': False,
            'required': ['visible', 'effects', 'intensity_mmi', 'basis'],
            'properties': {
                'visible': YNU,
                'effects': {'type': 'array', 'items': {'type': 'string'},
                            'description': 'visible effects of the shaking: swaying, falling objects, sloshing'},
                'intensity_mmi': {'type': 'string', 'description': 'a Modified Mercalli Intensity range from the '
                                  'visible effects, e.g., "V-VI"; empty if they do not constrain it'},
                'basis': {'type': 'string', 'description': 'the effects the intensity is based on'},
            }},
        **{key: _detection(what) for key, what in DETECTIONS.items()},
        'severity': {'type': 'integer', 'enum': [0, 1, 2, 3],
                     'description': '0 nothing notable; 1 shaking effects only; 2 possible damage, fire or ground '
                                    'failure (uncertain or minor); 3 clear damage, fire or ground failure'},
        'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'summary': {'type': 'string', 'description': 'one sentence for the map'},
    },
}

SEVERITY_NAMES = {0: 'nothing notable', 1: 'shaking effects only', 2: 'possible damage, fire or ground failure',
                  3: 'clear damage, fire or ground failure'}

SYSTEM_PROMPT = """You review frames from a public or traffic camera right after an earthquake, for a rapid \
survey of its effects. Responders use the survey to decide where to look first, so a missed fire or collapse \
costs more than a false alarm marked "uncertain", and a confident false alarm costs more than an honest \
"uncertain".

You get one image with up to four labeled frames from the same camera: BEFORE (just before the shaking), \
PEAK SHAKING, AFTER (once the shaking stopped), and LATEST (minutes later) or LATE SHAKING. You also get the \
camera's name and view, and the motion measured from the video: whether and how long it shook, how much the \
image moved, whether the camera moved or the image changed, and whether the video went dark or froze.

Compare the frames, above all BEFORE with AFTER and LATEST: what is new after the shaking is what matters. \
Report only what you can see. Things that often look like smoke or damage but are not: fog, haze, low sun, \
lens flare, rain or dirt on the lens, steam, traffic dust, motion blur and compression artifacts during the \
shaking, and a camera that panned or zoomed. A changed view is not damage unless you can see what changed.

Shaking intensity: estimate a Modified Mercalli Intensity range only from visible effects (for example, \
hanging objects or trees swaying, people running out, objects falling, damage), not from the motion of the \
image alone, which depends on how the camera is mounted.

People: describe only counts and actions (running, gathering, evacuating). Never describe or try to \
identify individuals.

Severity: 0 nothing notable; 1 shaking effects only; 2 possible damage, fire or ground failure (uncertain or \
minor); 3 clear damage, fire or ground failure."""


def user_text(task):
    """The text that goes with the image of one camera"""
    lines = [f"Camera: {task.get('camera_id')} - {task.get('name', '')}".rstrip(' -')]
    if task.get('view'):
        lines.append(f"View: {task['view']}")
    if task.get('distance_km') is not None:
        lines.append(f"Distance from the epicenter: {task['distance_km']} km")
    lines.append(f"Motion measured from the video: {task.get('motion_text', 'not available')}")
    lines.append('Review the frames and return the findings.')
    return '\n'.join(lines)


def validate(data, schema=FINDINGS_SCHEMA, path='findings'):
    """The problems of data against the schema (the subset of JSON Schema used here); empty if valid"""
    problems = []
    kind = schema.get('type')
    if kind == 'object':
        if not isinstance(data, dict):
            return [f'{path}: not an object']
        for key in schema.get('required', []):
            if key not in data:
                problems.append(f'{path}.{key}: missing')
        props = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            problems += [f'{path}.{key}: not expected' for key in data if key not in props]
        for key, sub in props.items():
            if key in data:
                problems += validate(data[key], sub, f'{path}.{key}')
        return problems
    if kind == 'array':
        if not isinstance(data, list):
            return [f'{path}: not an array']
        for k, item in enumerate(data):
            problems += validate(item, schema.get('items', {}), f'{path}[{k}]')
        return problems
    if kind == 'string' and not isinstance(data, str):
        problems.append(f'{path}: not a string')
    if kind == 'integer' and (not isinstance(data, int) or isinstance(data, bool)):
        problems.append(f'{path}: not an integer')
    if 'enum' in schema and data not in schema['enum']:
        problems.append(f'{path}: {data!r} not one of {schema["enum"]}')
    return problems
