"""prepare: the motion of each camera's video, and the frames for review

Each video is decoded at a low resolution and a uniform frame rate to measure its motion: the differences of
consecutive frames, and the shift of the whole image relative to a calm frame (the camera shaking). From them,
the shaking (onset, peak, end), and the frames before, at the peak, after and the latest, saved at full
resolution with a labeled 2 x 2 tile of them for review. Cameras whose video is missing, edited (scene cuts),
ends dark or frozen, moved, or changed between before and after are flagged. With the event (origin time,
epicenter) and the time of the first frame of a video, the onset is searched near the predicted arrivals.
"""

import csv
import datetime as dt
import json
import math
import os
import subprocess

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft, ndimage

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

VP, VS = 6.0, 3.5                                   # km/s, for the predicted arrivals
RANK = {'high': 0, 'normal': 1, 'low': 2}


# ---------------------------------------------------------------------------------------------- inputs

def read_cameras(path):
    """The list of cameras and their videos: camera_id, name, lat, lon, video, and optionally start_utc
    (ISO time of the first frame) and view (what the camera looks at)"""
    with open(path, newline='') as f:
        rows = [{k.strip(): (v or '').strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
    for row in rows:
        for key in ('lat', 'lon'):
            row[key] = float(row[key]) if row.get(key) else None
    return rows


def parse_time(text):
    if not text:
        return None
    t = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance"""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


# ------------------------------------------------------------------------------------------- the video

def probe(video):
    """Width, height and duration of a video"""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height:format=duration', '-of', 'json', video],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    s = info['streams'][0]
    return int(s['width']), int(s['height']), float(info['format'].get('duration', 'nan'))


def decode_small(video, width, height, fps, start=0.0, duration=None):
    """Grayscale frames at a reduced size and a uniform frame rate (frames x height x width, uint8)"""
    trim = ['-ss', f'{start:.3f}'] if start > 0 else []
    trim += ['-t', f'{duration:.3f}'] if duration else []
    raw = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', *trim, '-i', video,
                          '-vf', f'fps={fps},scale={width}:{height}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width)


def extract_frame(video, t, out, duration=None):
    """One frame at full resolution at time t (s) in the video; near the end of a video, where a seek can
    return no frame, earlier by steps of 0.2 s"""
    if duration and duration == duration:                     # not nan
        t = min(t, duration - 0.1)
    if os.path.exists(out):
        os.remove(out)
    for step in range(6):
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{max(t - 0.2 * step, 0):.3f}',
                        '-i', video, '-frames:v', '1', '-q:v', '3', out], check=True)
        if os.path.exists(out):
            return out
    raise ValueError(f'no frame near {t:.2f} s in {video}')


# ------------------------------------------------------------------------------------------ the motion

def frame_differences(frames, chunk=200):
    """The mean absolute difference of consecutive frames"""
    d = np.empty(max(len(frames) - 1, 0), np.float32)
    for a in range(0, len(frames) - 1, chunk):
        b = min(a + chunk + 1, len(frames))
        d[a:b - 1] = np.abs(np.diff(frames[a:b].astype(np.float32), axis=0)).mean(axis=(1, 2))
    return d


def image_shifts(frames, ref, chunk=100):
    """The shift (down, across) of each frame relative to frame ref, by phase correlation, in pixels of the
    frames; sub-pixel from a parabola through the peak and its neighbors"""
    n, h, w = frames.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    r = frames[ref].astype(np.float32)
    R = np.conj(fft.fft2((r - r.mean()) * win))
    shifts = np.empty((n, 2), np.float32)
    for a in range(0, n, chunk):
        f = frames[a:a + chunk].astype(np.float32)
        f = (f - f.mean(axis=(1, 2), keepdims=True)) * win
        X = R * fft.fft2(f, axes=(1, 2), workers=-1)
        c = fft.ifft2(X / (np.abs(X) + 1e-6), axes=(1, 2), workers=-1).real
        for k in range(len(c)):
            iy, ix = np.unravel_index(np.argmax(c[k]), (h, w))
            sub = []
            for i, size, line in ((iy, h, c[k][:, ix]), (ix, w, c[k][iy])):
                cm, c0, cp = line[(i - 1) % size], line[i], line[(i + 1) % size]
                den = cm - 2 * c0 + cp
                frac = 0.5 * (cm - cp) / den if den < 0 else 0.0
                sub.append(((i + frac + size / 2) % size) - size / 2)
            shifts[a + k] = sub
    return shifts


def find_shaking(d, fps, lo, hi, calm_s=2.0, lead_s=0.5):
    """The onset, peak and end of the shaking (frame indices) from the frame differences d, searched in frames
    lo-hi; the threshold is 10 times the median difference of the first calm frames (after any leading freeze)"""
    frozen = d < 0.05
    run = ndimage.median_filter(d, size=5, mode='nearest') if len(d) else d
    first_live = 0
    if len(d) and frozen[0] and not frozen.all():
        k = int(np.argmax(~frozen))
        # a leading freeze (a re-upload) resumes calmly; identical frames followed at once by motion are a
        # static, noiseless scene whose calm frames are the frozen ones
        first_live = k if run[k:k + 5].max() < 2.0 else 0
    calm = d[first_live:first_live + max(5, int(calm_s * fps))]
    thr = max(0.5, 10 * float(np.median(calm))) if len(calm) else 0.5
    lo, hi = max(lo, first_live), min(hi, len(d))
    above = run[lo:hi] > thr
    if not above.any():
        return dict(detected=False, threshold=thr, first_live=first_live)
    onset = lo + int(np.argmax(above)) + 1
    end = lo + int(np.flatnonzero(above)[-1]) + 1
    smooth = ndimage.uniform_filter1d(d, 3)
    peak = onset + int(np.argmax(smooth[onset - 1:end]))
    ref = max(first_live, onset - max(1, int(lead_s * fps)))
    return dict(detected=True, threshold=thr, first_live=first_live, onset=onset, peak=peak, end=end, ref=ref,
                calm_before=bool(onset - first_live > calm_s * fps))


def changed_fraction(a, b, shift, level=25.0):
    """The fraction of the image that changed between frames a and b (b aligned onto a by shift), after
    smoothing; a margin of 5 % is left out"""
    fa = ndimage.gaussian_filter(a.astype(np.float32), 1.5)
    fb = ndimage.shift(ndimage.gaussian_filter(b.astype(np.float32), 1.5), -np.asarray(shift), order=1,
                       mode='nearest')
    h, w = a.shape
    my, mx = max(1, h // 20), max(1, w // 20)
    return float((np.abs(fa - fb)[my:-my, mx:-mx] > level).mean())


# --------------------------------------------------------------------------------------- the products

def tile(paths, labels, out, size=(640, 360)):
    """A 2 x 2 tile of up to four frames, each fitted into size and labeled"""
    canvas = Image.new('RGB', (2 * size[0], 2 * size[1]), 'black')
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for k, (path, label) in enumerate(zip(paths, labels)):
        if not path:
            continue
        with Image.open(path) as im:
            im = im.convert('RGB')
            im.thumbnail(size)
            x0 = (k % 2) * size[0] + (size[0] - im.width) // 2
            y0 = (k // 2) * size[1] + (size[1] - im.height) // 2
            canvas.paste(im, (x0, y0))
        box = draw.textbbox((x0 + 8, y0 + 6), label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 6), label, fill=(255, 220, 0), font=font)
    canvas.save(out, quality=90)
    return out


def plot_motion(t, d, shift_px, sh, marks, title, out):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(t[1:], d, color='#4a6fa5', lw=1)
    axes[0].axhline(sh['threshold'], color='#d62728', lw=0.8, ls='--', label='threshold of the shaking')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('frame difference', fontsize=9)
    axes[0].set_title(title, fontsize=10, loc='left')
    if shift_px is not None:
        axes[1].plot(t, shift_px[:, 1], lw=1, label='across')
        axes[1].plot(t, shift_px[:, 0], lw=1, label='down')
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel('image shift, px', fontsize=9)
    axes[1].set_xlabel('time in the video, s', fontsize=9)
    for ax in axes:
        for name, tm in marks.items():
            if tm is not None:
                ax.axvline(tm, color='#555555', lw=0.7, ls=':')
                if ax is axes[0]:
                    ax.annotate(name, (tm, 1), xycoords=('data', 'axes fraction'), xytext=(2, -10),
                                textcoords='offset points', fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def motion_text(m):
    """The motion of a camera in words, for the review"""
    if 'no_video' in m['flags']:
        return 'No video for this camera.'
    parts = []
    if m.get('shaking_detected'):
        parts.append(f"Shaking from {m['onset_s']} s to {m['end_s']} s in the video (above the threshold for "
                     f"{m['strong_duration_s']} s), peak at {m['peak_s']} s; image shake about "
                     f"{m['shake_amplitude_px']} px, peak image velocity {m['peak_image_velocity_px_s']} px/s, "
                     f"permanent shift {m['permanent_shift_px']} px.")
    elif 'shaking_detected' in m:
        parts.append('No shaking detected in the video.')
    for key in ('changed_fraction_after', 'changed_fraction_latest'):
        if key in m:
            parts.append(f"Image changed between BEFORE and {key.rsplit('_', 1)[1].upper()}: "
                         f"{100 * m[key]:.1f} % of the area.")
    if m['flags']:
        parts.append('Flags: ' + ', '.join(m['flags']) + '.')
    return ' '.join(parts)


# ------------------------------------------------------------------------------------------- prepare

def prepare_camera(cam, videos, outdir, event, motion_fps=15.0, motion_width=320, max_seconds=600.0):
    cid = cam['camera_id']
    cdir = os.path.join(outdir, 'cameras', cid)
    os.makedirs(cdir, exist_ok=True)
    m = dict(camera_id=cid, name=cam.get('name', ''), lat=cam.get('lat'), lon=cam.get('lon'),
             view=cam.get('view', ''), flags=[])
    if event and event.get('lat') is not None and cam.get('lat') is not None:
        m['distance_km'] = round(distance_km(event['lat'], event['lon'], cam['lat'], cam['lon']), 1)
    video = os.path.join(videos, cam['video']) if cam.get('video') else ''
    if not video or not os.path.exists(video):
        m['flags'].append('no_video')
        m['priority'] = 'high'
        return m
    width, height, duration = probe(video)
    m.update(video=os.path.abspath(video), width=width, height=height, duration_s=round(duration, 2))

    # the window of the predicted arrivals, if the time of the first frame is known
    start, lo_s, hi_s = 0.0, 0.0, None
    t0 = parse_time(cam.get('start_utc'))
    if t0 and event and event.get('origin_utc') and 'distance_km' in m:
        hypo = math.hypot(m['distance_km'], event.get('depth_km') or 10.0)
        origin = (parse_time(event['origin_utc']) - t0).total_seconds()
        m['predicted_p_s'], m['predicted_s_s'] = round(origin + hypo / VP, 1), round(origin + hypo / VS, 1)
        lo_s, hi_s = m['predicted_p_s'] - 30, m['predicted_s_s'] + 120
        if hi_s < 0 or lo_s > duration:
            m['flags'].append('predicted_arrivals_outside_the_video')
            lo_s, hi_s = 0.0, None
    if duration > max_seconds:                    # decode only the part that matters
        start = max(0.0, lo_s - 30) if hi_s is not None else 0.0
        m['flags'].append(f'decoded {max_seconds:.0f} s of {duration:.0f} s from {start:.0f} s')

    h_small = max(2, int(round(height * motion_width / width / 2)) * 2)
    frames = decode_small(video, motion_width, h_small, motion_fps, start, max_seconds)
    n = len(frames)
    if n < 3 * motion_fps:
        m['flags'].append('too_short')
    if n < 2:
        raise ValueError(f'{n} frames decoded')
    t = start + np.arange(n) / motion_fps
    d = frame_differences(frames)
    cuts = np.flatnonzero((d > 30) & (d > 8 * ndimage.median_filter(d, size=9, mode='nearest')))
    if len(cuts):
        m['flags'].append(f'scene_cuts ({len(cuts)}): an edited video, the motion is not reliable')
    lo = int((lo_s - start) * motion_fps) if lo_s else 0
    hi = int((hi_s - start) * motion_fps) if hi_s is not None else n
    sh = find_shaking(d, motion_fps, max(lo, 0), max(hi, 1))
    scale = width / motion_width
    m['shaking_detected'] = sh['detected']
    m['threshold'] = round(sh['threshold'], 2)

    brightness = frames.reshape(n, -1).mean(axis=1)
    tail = max(1, int(2 * motion_fps))
    shifts = shift_px = None
    if sh['detected']:
        ref, onset, peak, end = sh['ref'], sh['onset'], sh['peak'], sh['end']
        shifts = image_shifts(frames, ref)
        shift_px = shifts * scale
        before = shift_px[max(sh['first_live'], ref - tail):ref + 1]
        during = shift_px[onset:end + 1]
        after_i = min(n - 1, end + int(motion_fps))
        settled = shift_px[after_i:min(n, after_i + tail)]
        vel = np.linalg.norm(np.diff(ndimage.uniform_filter1d(shift_px, 3, axis=0), axis=0), axis=1) * motion_fps
        permanent = float(np.linalg.norm(np.median(settled, 0) - np.median(before, 0))) if len(settled) else None
        m.update(onset_s=round(float(t[onset]), 2), peak_s=round(float(t[peak]), 2), end_s=round(float(t[end]), 2),
                 strong_duration_s=round((end - onset) / motion_fps, 2),
                 shake_amplitude_px=round(float(np.max(np.ptp(during, axis=0)) / 2), 1),
                 peak_image_velocity_px_s=round(float(vel[onset - 1:end].max()), 1),
                 permanent_shift_px=None if permanent is None else round(permanent, 1),
                 calm_frames_before=sh['calm_before'])
        if not sh['calm_before']:
            m['flags'].append('no_calm_frames_before_the_shaking')
        if end >= n - 2:
            m['flags'].append('shaking_until_the_end_of_the_clip')
        if permanent and permanent > 3:
            m['flags'].append('camera_moved')
        picks = dict(before=ref, peak=peak, after=after_i, latest=n - 1)
    else:
        ref = sh['first_live']
        picks = dict(before=ref, peak=None, after=None, latest=n - 1)

    # the end of the video: dark or frozen
    if brightness[-tail:].mean() < max(20.0, 0.3 * brightness[ref]):
        m['flags'].append('dark_at_the_end')
    if n > 2 * tail and (d[-tail:] < 0.05).all() and not (d[:tail] < 0.05).all():
        m['flags'].append('frozen_at_the_end')

    # the change between before and after, and before and the latest frame
    for name in ('after', 'latest'):
        k = picks[name]
        if k is not None and k != ref:
            s = shifts[k] if shifts is not None else np.zeros(2)
            m[f'changed_fraction_{name}'] = round(changed_fraction(frames[ref], frames[k], s), 4)
    if max(m.get('changed_fraction_after', 0), m.get('changed_fraction_latest', 0)) > 0.03:
        m['flags'].append('scene_changed')

    # the frames at full resolution, and the tile
    if picks['after'] is not None and picks['latest'] - picks['after'] < 2 * motion_fps:
        mid = (sh['peak'] + sh['end']) // 2
        picks['latest'] = None if mid == sh['peak'] else mid
        order = [('before', 'BEFORE'), ('peak', 'PEAK SHAKING'), ('latest', 'LATE SHAKING'), ('after', 'AFTER')]
    else:
        order = [('before', 'BEFORE'), ('peak', 'PEAK SHAKING'), ('after', 'AFTER'), ('latest', 'LATEST')]
    m['frames'], paths, labels = {}, [], []
    for key, label in order:
        k = picks[key]
        if k is None:
            paths.append(None)
            labels.append('')
            continue
        tk = float(t[k])
        path = extract_frame(video, tk, os.path.join(cdir, f'{key}.jpg'), duration)
        when = (t0 + dt.timedelta(seconds=tk)).strftime('%H:%M:%S UTC') if t0 else f'{tk:.1f} s'
        m['frames'][key] = dict(file=os.path.relpath(path, outdir), video_time_s=round(tk, 2))
        paths.append(path)
        labels.append(f'{label}  {when}')
    m['tile'] = os.path.relpath(tile(paths, labels, os.path.join(cdir, 'tile.jpg')), outdir)
    marks = {key: float(t[picks[key]]) if picks.get(key) is not None else None for key, _ in order}
    plot_motion(t, d, shift_px, sh, marks, f'{cid}: {m["name"]}', os.path.join(cdir, 'motion.png'))
    m['motion_plot'] = os.path.relpath(os.path.join(cdir, 'motion.png'), outdir)
    np.savez_compressed(os.path.join(cdir, 'motion.npz'), t=t, d=d, brightness=brightness,
                        shift_px=shift_px if shift_px is not None else np.zeros((0, 2)))

    strong = sh['detected'] and (m['shake_amplitude_px'] > 2 or m['peak_image_velocity_px_s'] > 20)
    alarm = {'dark_at_the_end', 'frozen_at_the_end', 'scene_changed', 'camera_moved'} & set(m['flags'])
    m['priority'] = 'high' if alarm else ('normal' if strong else 'low')
    return m


def prepare(cameras_csv, videos, out, event_json=None, only=None, **options):
    """Prepare every camera; writes OUT/metrics.json, OUT/tasks.json and OUT/event.json"""
    event = {}
    if event_json:
        with open(event_json) as f:
            event = json.load(f)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'event.json'), 'w') as f:
        json.dump(event, f, indent=1)
    cams = read_cameras(cameras_csv)
    if only:
        cams = [c for c in cams if c['camera_id'] in only]
    metrics = []
    for cam in cams:
        try:
            m = prepare_camera(cam, videos, out, event, **options)
        except (subprocess.CalledProcessError, ValueError, IndexError, OSError) as err:
            m = dict(camera_id=cam['camera_id'], name=cam.get('name', ''), lat=cam.get('lat'), lon=cam.get('lon'),
                     view=cam.get('view', ''), flags=[f'error: {type(err).__name__}: {err}'], priority='high')
        m['motion_text'] = motion_text(m)
        metrics.append(m)
        print(f"{m['camera_id']:>20s}  {m['priority']:6s}  {m['motion_text']}")
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    keys = ('camera_id', 'name', 'view', 'distance_km', 'priority', 'tile', 'frames', 'motion_plot', 'motion_text')
    tasks = [{k: m.get(k) for k in keys} for m in sorted(metrics, key=lambda m: RANK[m['priority']])]
    with open(os.path.join(out, 'tasks.json'), 'w') as f:
        json.dump(tasks, f, indent=1)
    print(f'{len(tasks)} cameras -> {out}/tasks.json')
    return tasks
