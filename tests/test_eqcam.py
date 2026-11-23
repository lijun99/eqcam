"""Tests without any API: a synthetic video with shaking and a change after it, the schema, the request of each
backend (a fake Anthropic client, a local stand-in for an OpenAI-compatible server), and review and report with
a fake backend"""

import csv
import glob
import json
import math
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from eqcam import arrivals, backends, calibrate, frames, mask, report, review, schema, vibration

FPS, H, W = 30, 360, 640


def example_findings(severity=2):
    fd = {key: {'present': 'no', 'description': '', 'where': ''} for key in schema.DETECTIONS}
    fd.update(camera_status='ok', severity=severity, confidence='medium', summary='Smoke after the shaking.',
              shaking={'visible': 'yes', 'effects': ['poles swaying'], 'intensity_mmi': 'V-VI', 'mmi_kind': 'estimate',
                       'mmi_confidence': 60,
                       'basis': 'poles and wires swaying'})
    fd['smoke_or_dust'] = {'present': 'yes', 'description': 'a plume', 'where': 'upper left, AFTER'}
    fd['people_boxes'] = [{'x_min': 100, 'y_min': 100, 'x_max': 200, 'y_max': 300}]
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
    assert 'view_changes_s' not in cam1 and any(7.5 < a <= 8.3 for a, _ in cam1['sudden_changes_s'])
    assert 'predicted_p_s' in cam1                            # start_utc and the event are known
    with Image.open(os.path.join(prepared, cam1['tile'])) as im:
        assert im.size == (1280, 720)
    assert set(cam1['frames']) == {'before', 'peak', 'after', 'latest'}


def test_shaking_of_a_noiseless_static_scene(tmp_path):
    """Identical calm frames followed by motion are a static scene, not a frozen upload"""
    write_video(str(tmp_path / 'still.mp4'), seconds=8, shake=(3.0, 6.0), change_at=100, noise=0)
    frames_small = frames.decode_small(str(tmp_path / 'still.mp4'), 320, 180, 15)
    sh = frames.analyze_motion(frames_small, 15)
    assert sh['detected'] and abs(sh['onset'] / 15 - 3.0) < 0.5 and not sh['changes']


def write_tour_video(path, shake=(2.0, 4.0), seconds=12):
    """A camera touring its views: view A (shaken between shake[0] and shake[1] s), a pan to the right from 5 to
    7 s, a cut to another scene at 7 s, and a cut back to view A at 9.5 s"""
    rng = np.random.default_rng(3)
    scenes = []
    for _ in range(2):
        sc = ndimage.gaussian_filter(rng.uniform(0, 255, (H + 40, W + 240)), 3)
        scenes.append((255 * (sc - sc.min()) / np.ptp(sc)).astype(np.uint8))
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(int(seconds * FPS)):
        t = k / FPS
        scene, x, dy, dx = scenes[0], 20, 0, 0
        if shake[0] <= t < shake[1]:
            dy, dx = int(round(4 * np.sin(2 * np.pi * 3 * t))), int(round(5 * np.cos(2 * np.pi * 2.5 * t)))
        if 5 <= t < 7:
            x = 20 + int(round(90 * (t - 5)))                    # 90 px/s
        elif 7 <= t < 9.5:
            scene = scenes[1]
        img = scene[20 + dy:20 + dy + H, x + dx:x + dx + W].astype(np.int16)
        img += rng.integers(-3, 4, img.shape, dtype=np.int16)
        proc.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def test_camera_tour_is_not_shaking(tmp_path):
    """A pan and cuts between views are view changes, not shaking; the shaking is found within its view"""
    write_tour_video(str(tmp_path / 'tour.mp4'), shake=(100, 100))
    sh = frames.analyze_motion(frames.decode_small(str(tmp_path / 'tour.mp4'), 320, 180, 15), 15)
    assert not sh['detected']
    starts = [a / 15 for a, _ in sh['changes']]
    assert any(3.8 < s < 5.5 for s in starts)                                   # the pan from 5 s
    assert all(any(a / 15 - 0.2 < tc < b / 15 + 0.2 for a, b in sh['changes']) for tc in (6, 7, 9.5))


def test_shaking_in_a_touring_camera(tmp_path):
    """The shaking in view A; BEFORE, PEAK and AFTER in view A, and LATEST when the camera is back in view A"""
    write_tour_video(str(tmp_path / 'tour.mp4'))
    with open(tmp_path / 'cameras.csv', 'w', newline='') as f:
        csv.writer(f).writerows([['camera_id', 'name', 'lat', 'lon', 'video'], ['tour', 'Tour', 34, -118, 'tour.mp4']])
    frames.prepare(str(tmp_path / 'cameras.csv'), str(tmp_path), str(tmp_path / 'out'))
    m = metrics_of(str(tmp_path / 'out'))['tour']
    assert m['shaking_detected'] and abs(m['onset_s'] - 2.0) < 0.5 and abs(m['end_s'] - 4.0) < 0.7
    t = {k: v['video_time_s'] for k, v in m['frames'].items()}
    assert t['before'] < 2.1 and 2 < t['peak'] < 4 and 4 <= t['after'] < 5 and t['latest'] > 10
    assert 'latest_in_another_view' not in m['flags'] and 'scene_changed' not in m['flags']
    assert len(m['view_changes_s']) >= 2 and 'not the earthquake' in m['motion_text']


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


def test_map_tags():
    """Every detection of the schema is a filter of the map, with a label; tags from the findings and flags"""
    assert set(schema.DETECTIONS) <= set(report.TAGS) and set(report.TAGS) <= set(report.TAG_LABELS)
    fd = {'shaking': {'visible': 'yes'}, 'fire': {'present': 'uncertain'}, 'water': {'present': 'no'},
          'camera_status': 'ok'}
    assert report.tags({'findings': fd, 'flags': []}) == {'shaking': 'yes', 'fire': 'uncertain'}
    assert report.tags({'findings': {}, 'flags': ['no_video']}) == {'camera_problem': 'yes'}
    assert report.tags({'findings': {'camera_status': 'dark'}}) == {'camera_problem': 'yes'}


def test_video_window_and_links():
    """The window of the video: set by the review, else the camera vibration detected, else the predicted arrivals"""
    r = dict(video='/data/videos/cam1.mp4', duration_s=60.0, predicted_p_s=3.0, predicted_s_s=6.0, shaking_detected=False,
             frames={'before': {'video_time_s': 1.5}}, record={})
    assert report.video_window(r) == (0.0, 36.0, 'predicted P and S arrivals')
    r.update(shaking_detected=True, onset_s=8.0, end_s=58.0)
    assert report.video_window(r) == (3.0, 60.0, 'camera vibration detected')
    r['record'] = {'video_window_s': [10, 20]}
    assert report.video_window(r) == (10, 20, 'set by the review')
    links = report.video_links(dict(r, video_url='https://example.org/v.mp4'))
    assert 'https://example.org/v.mp4#t=10,20' in links and '#t=1.5"' in links and 'BEFORE 1.5 s' in links
    assert 'cam1.mp4' not in links                       # never the local file
    assert report.video_links(r) == ''


def test_motion_items():
    """The motion in short statements"""
    m = dict(flags=['view_changes (3)', 'scene_changed', 'dark_at_the_end'], shaking_detected=True, onset_s=7.0,
             end_s=25.4, strong_duration_s=18.4, shake_amplitude_px=6.0, view_changes_s=[[5, 6], [25, 26], [45, 46]],
             changed_fraction_after=0.042)
    items = frames.motion_items(m)
    assert items[0] == 'Camera vibration 7.0-25.4 s (18 s), up to 6 px'
    assert 'Camera pans or switches views 3 times (not the earthquake)' in items
    assert 'Scene changed: 4 % of the image' in items and 'Dark at the end (possible outage)' in items
    assert frames.motion_items(dict(flags=['no_video'])) == ['No video']
    assert frames.motion_items(dict(flags=[], shaking_detected=False)) == ['No camera vibration detected']


def test_mask_people(prepared, tmp_path):
    """The boxes of the review blurred in the tile and the mapped part of the frame; the originals kept, and
    restored when the boxes are gone"""
    out = str(tmp_path / 'run')
    shutil.copytree(prepared, out)
    shutil.rmtree(os.path.join(out, 'findings'), ignore_errors=True)                 # other tests reviewed it
    orig = os.path.join(out, 'cameras', 'cam1', 'original')
    if os.path.isdir(orig):                                                         # and masked it
        for name in os.listdir(orig):
            shutil.copy2(os.path.join(orig, name), os.path.join(out, 'cameras', 'cam1', name))
        shutil.rmtree(orig)
    m = metrics_of(out)['cam1']
    assert set(m['tile_layout']) == set(m['frames'])
    tile_path, before_path = os.path.join(out, m['tile']), os.path.join(out, m['frames']['before']['file'])
    tile0, before0 = (np.asarray(Image.open(p).convert('L'), float) for p in (tile_path, before_path))
    rec = dict(camera_id='cam1', stage='review', findings=dict(example_findings(), people_boxes=[
        {'x_min': 100, 'y_min': 100, 'x_max': 300, 'y_max': 400}]))      # in the BEFORE frame (upper left)
    os.makedirs(os.path.join(out, 'findings', 'review'))
    with open(os.path.join(out, 'findings', 'review', 'cam1.json'), 'w') as f:
        json.dump(rec, f)
    assert mask.mask_run(out)['cam1'] == 1
    tile1, before1 = (np.asarray(Image.open(p).convert('L'), float) for p in (tile_path, before_path))
    inside = (slice(int(0.45 * 720 * 0.4), int(0.35 * 720)), slice(140, 380))         # within the box, in the tile
    assert np.abs(tile1 - tile0)[inside].mean() > 5                                   # blurred
    assert np.abs(tile1 - tile0)[400:, 700:].mean() < 1                               # LATEST, untouched
    x0, y0, w, h = m['tile_layout']['before']
    fy, fx = before0.shape[0] / h, before0.shape[1] / w
    box = (slice(int((144 - y0) * fy), int((216 - y0) * fy)), slice(int((160 - x0) * fx), int((320 - x0) * fx)))
    assert np.abs(before1 - before0)[box].mean() > 5 and np.abs(before1 - before0)[-50:, -50:].mean() < 1
    assert os.path.exists(os.path.join(out, 'cameras', 'cam1', 'original', 'tile.jpg'))
    rec['findings']['people_boxes'] = []                                              # a new review: no people
    with open(os.path.join(out, 'findings', 'review', 'cam1.json'), 'w') as f:
        json.dump(rec, f)
    mask.mask_run(out)
    assert np.abs(np.asarray(Image.open(tile_path).convert('L'), float) - tile0).mean() < 1


def test_mount_response():
    """A lightly damped ringing at one frequency is a resonant mount; a short broadband burst is not; jumps are
    not measured"""
    fps, onset = 30.0, 60
    t = np.arange(int(30 * fps)) / fps
    rng = np.random.default_rng(0)
    after = np.clip(t - onset / fps, 0, None) * (t >= onset / fps)
    ring = 2.0 * np.exp(-0.01 * 2 * np.pi * 4 * after) * np.sin(2 * np.pi * 4 * after)        # 4 Hz, 1 % damping
    shifts = np.stack([ring, 0.3 * ring], 1) + rng.normal(0, 0.005, (len(t), 2))
    m = frames.mount_response(shifts, fps, onset)
    assert m['resonant'] and abs(m['frequency_hz'] - 4) < 0.2 and 2.5 < m['ring_s'] < 6 and m['concentration'] > 0.7
    burst = rng.normal(0, 1, (len(t), 2)) * np.exp(-0.5 * ((t - 4) / 0.7) ** 2)[:, None]      # broadband, 1-2 s
    m = frames.mount_response(burst + rng.normal(0, 0.005, (len(t), 2)), fps, onset)
    assert not m['resonant'] and m['concentration'] < 0.5
    jump = shifts.copy()
    jump[300:] += 50
    assert frames.mount_response(jump, fps, onset) is None
    assert 'resonates' in frames.mount_text(dict(frequency_hz=4.0, concentration=0.9, ring_s=4.0, damping=0.01,
                                                 ringing_to_end=False, resonant=True))


def test_vibration_finds_an_object(tmp_path):
    """A camera ringing at 5 Hz, and an object in its view swinging on its own at 2.2 Hz: the camera's frequency,
    and the object, where it is and at its frequency; nothing else"""
    rng = np.random.default_rng(5)
    scene = ndimage.gaussian_filter(rng.uniform(0, 255, (H + 40, W + 40)), 2)
    scene = (255 * (scene - scene.min()) / np.ptp(scene)).astype(np.float32)
    path = str(tmp_path / 'vib.mp4')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(int(16 * FPS)):
        t = k / FPS
        cam = 1.5 * np.exp(-0.3 * (t - 6)) * np.sin(2 * np.pi * 5 * (t - 6)) if t >= 6 else 0.0
        obj = 2.0 * np.sin(2 * np.pi * 2.2 * (t - 6)) if 6 <= t < 14 else 0.0
        img = ndimage.shift(scene, (cam, 0.6 * cam), order=1)[20:20 + H, 20:20 + W]
        moved = ndimage.shift(scene, (cam + 0.5 * obj, 0.6 * cam + obj), order=1)[20:20 + H, 20:20 + W]
        img[100:200, 300:420] = moved[100:200, 300:420]                   # the object
        proc.stdin.write(np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    out = str(tmp_path / 'run')
    os.makedirs(os.path.join(out, 'cameras', 'vib'))
    m = dict(camera_id='vib', name='synthetic', shaking_detected=True, video=path, onset_s=6.0, end_s=12.0,
             duration_s=16.0, width=W, height=H, frames={}, view_changes_s=[], distance_km=10.0)
    v = vibration.analyze_camera(out, m)
    assert abs(v['camera']['frequency_hz'] - 5) < 0.3
    assert len(v['objects']) >= 1 and all(abs(o['frequency_hz'] - 2.2) < 0.3 for o in v['objects'])
    x0, y0, x1, y1 = v['objects'][0]['box']
    assert x0 < 420 and x1 > 300 and y0 < 200 and y1 > 100                  # where the object is
    assert os.path.exists(os.path.join(out, v['plot']))
    rows = [dict(m, lat=34.0, lon=-118.0, vibration=v), dict(camera_id='quiet', lat=34.1, lon=-118.1)]
    vibration.summary_png(rows, dict(event_id='test', lat=34.05, lon=-118.05), out)
    vibration.map_html(rows, dict(event_id='test', lat=34.05, lon=-118.05), out)
    assert os.path.exists(os.path.join(out, 'vibration.html'))


def test_arrivals():
    """A weak 8 Hz P and a strong S 3.0 s later (the expected S-P), behind a stream delay: P and S, the delay from
    S, P on time; a one-frame glitch does not count; P 0.7 s late (S-P short): still P and S, noted late; without
    the P, S only"""
    fps, rng = 30.0, np.random.default_rng(2)
    t = np.arange(int(25 * fps)) / fps
    burst = lambda t0, a, f, tau: a * np.where(t >= t0, np.exp(-(t - t0) / tau), 0) * np.sin(2 * np.pi * f * (t - t0))  # noqa: E731
    x = np.stack([burst(10, 0.05, 8, 1.5) + burst(13, 0.6, 3, 3), burst(10, 0.03, 7, 1.5) + burst(13, 0.4, 2.5, 3)], 1)
    x += rng.normal(0, 0.002, x.shape)
    x[150] += 0.8                                                        # a glitch at 5 s
    hypo = 3.0 / (1 / frames.VS - 1 / frames.VP)
    m = dict(camera_id='c', distance_km=math.sqrt(hypo ** 2 - 100), predicted_p_s=hypo / frames.VP,
             predicted_s_s=hypo / frames.VS, flags=[])
    p, _, _ = arrivals.pick(t, x, fps, m, dict(depth_km=10))
    assert p['quality'] == 'P and S' and abs(p['p_s'] - 10) < 0.2 and abs(p['s_s'] - 13) < 0.2
    assert abs(p['delay_s_s'] - (13 - m['predicted_s_s'])) < 0.2 and abs(p['p_late_s']) < 0.3 and not p['notes']
    late = np.stack([burst(10.7, 0.05, 8, 1.5) + burst(13, 0.6, 3, 3), burst(10.7, 0.03, 7, 1.5) + burst(13, 0.4, 2.5, 3)], 1)
    p, _, _ = arrivals.pick(t, late + rng.normal(0, 0.002, x.shape), fps, m, dict(depth_km=10))
    assert p['quality'] == 'P and S' and abs(p['p_late_s'] - 0.7) < 0.2 and 'later than the S delay' in p['notes'][0]
    x[:, :] = np.stack([burst(13, 0.6, 3, 3), burst(13, 0.4, 2.5, 3)], 1) + rng.normal(0, 0.002, x.shape)
    p, _, _ = arrivals.pick(t, x, fps, m, dict(depth_km=10))
    assert p['quality'] == 'S only' and abs(p['s_s'] - 13) < 0.2 and p['p_s'] is None


def test_calibrate(tmp_path, monkeypatch):
    """A clip of a camera swaying 0.2 px at 4 Hz in the wind, and a dark, frozen clip of the same camera by
    night; an earlier event that measured its mount at 4 Hz and a stream delay: the catalog has the ambient
    peak, matching the event, the bad night clip, and the delay"""
    assert calibrate.sun_elevation(frames.parse_time('2026-11-20T11:25:00Z'), 34.05, -118.25) < -6   # 3:25 am
    assert calibrate.sun_elevation(frames.parse_time('2026-06-21T20:00:00Z'), 34.05, -118.25) > 70   # 1 pm
    rng = np.random.default_rng(7)
    scene = ndimage.gaussian_filter(rng.uniform(0, 255, (H + 40, W + 40)), 2)
    scene = (255 * (scene - scene.min()) / np.ptp(scene)).astype(np.float32)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p']
    proc = subprocess.Popen(cmd + [str(tmp_path / 'windy.mp4')], stdin=subprocess.PIPE)
    for k in range(int(30 * FPS)):
        sway = 0.2 * np.sin(2 * np.pi * 4.0 * k / FPS + 0.3 * np.sin(k / FPS))
        img = ndimage.shift(scene, (sway, 0.5 * sway), order=1)[20:20 + H, 20:20 + W]
        proc.stdin.write(np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    proc = subprocess.Popen(cmd + [str(tmp_path / 'night.mp4')], stdin=subprocess.PIPE)
    dark = (0.05 * scene[20:20 + H, 20:20 + W]).astype(np.uint8).tobytes()
    for k in range(int(10 * FPS)):
        proc.stdin.write(dark)
    proc.stdin.close()
    assert proc.wait() == 0
    with open(tmp_path / 'clips.csv', 'w') as f:
        f.write('camera_id,name,lat,lon,video,start_utc\n'
                'c1,pole,34.05,-118.25,windy.mp4,2026-06-21T20:00:00Z\n'
                'c1,pole,34.05,-118.25,night.mp4,2026-11-20T11:25:00Z\n')
    run = tmp_path / 'event' / 'run'
    os.makedirs(run / 'vibration_data')
    (run / 'event.json').write_text(json.dumps(dict(event_id='ev1', lat=34.0, lon=-118.2, magnitude=4.0)))
    (run / 'metrics.json').write_text(json.dumps([dict(
        camera_id='c1', name='pole', lat=34.05, lon=-118.25, shaking_detected=True,
        mount=dict(frequency_hz=4.1, damping=0.02, resonant=True))]))
    (run / 'vibration_data' / 'cameras.json').write_text(json.dumps([dict(
        camera_id='c1', picks=dict(quality='P and S', delay_p_s=2.0, delay_s_s=2.2))]))
    out = str(tmp_path / 'catalog')
    monkeypatch.setitem(calibrate.QUALITY, 'min_height_px', 360)       # the synthetic video is 640 x 360
    calibrate.calibrate(str(tmp_path / 'clips.csv'), str(tmp_path), out, [str(run)])
    (c,) = json.load(open(os.path.join(out, 'catalog.json')))['cameras']
    assert c['day']['usable'] and not c['day']['problems']
    assert not c['night']['usable'] and {'dark', 'frozen'} <= set(c['night']['problems'])
    assert c['touring'] is False and c['vibration_usable']
    assert abs(c['ambient']['frequency_hz'] - 4.0) < 0.2
    assert 'an ambient peak' in c['notes'][0]
    assert c['stream_delay_s'] == dict(median=2.2, n=1) and c['resonant_at_events']
    assert os.path.exists(os.path.join(out, c['figure']))
    rows = list(csv.DictReader(open(os.path.join(out, 'catalog.csv'))))
    assert rows[0]['ambient_frequency_hz'] == '4.0' and rows[0]['stream_delay_s'] == '2.2'
    # the periodic cycle: record each stream (here a file, which ffmpeg reads as it would a stream; and one
    # that does not exist), calibrate, and report the changes; then the first camera goes offline
    streams = tmp_path / 'streams.csv'
    streams.write_text('camera_id,name,lat,lon,stream\n'
                       f"c1,pole,34.05,-118.25,{tmp_path / 'windy.mp4'}\n"
                       f"c2,gone,34.06,-118.26,{tmp_path / 'missing.mp4'}\n")
    calibrate.calibrate(None, None, out, streams=str(streams), record_s=20)
    cams = {c['camera_id']: c for c in json.load(open(os.path.join(out, 'catalog.json')))['cameras']}
    assert cams['c1']['clips'] == 3 and cams['c1']['uptime'] == 1.0
    assert cams['c2']['uptime'] == 0.0 and cams['c2']['latest_problems'] == ['offline']
    assert any('c2 gone: new' in c for c in json.load(open(os.path.join(out, 'changes.json')))['changes'])
    (recorded,) = glob.glob(os.path.join(out, 'recordings', '*', 'c1.mp4'))
    time.sleep(1.1)                                             # the next run in its own folder
    streams.write_text('camera_id,name,lat,lon,stream\n'
                       f"c1,pole,34.05,-118.25,{tmp_path / 'missing.mp4'}\n")
    calibrate.calibrate(None, None, out, streams=str(streams), record_s=20, keep_days=0)
    changes = json.load(open(os.path.join(out, 'changes.json')))['changes']
    assert changes == ['c1 pole: offline at the latest clip']
    assert not os.path.exists(recorded)                         # pruned (keep_days=0)
