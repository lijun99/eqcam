"""prepare: the motion of each camera's video, and the frames for review

Each video is decoded at a low resolution and a uniform frame rate to measure its motion by phase correlation:
the shift of each frame relative to the one before it and to the one 1 s before it. From them, the views: the
camera panning, tilting, zooming or switching views (a steady drift of the image, or frames that do not match)
separates stable views; and the shaking: the whole image jittering back and forth within a stable view (onset,
peak, end). The frames before, at the peak and after the shaking are taken in its view, and the latest frame in
the same view if the camera comes back to it; they are saved at full resolution with a labeled 2 x 2 tile for
review. Cameras whose video is missing, ends dark or frozen, moved, or changed between before and after are
flagged. With the event (origin time, epicenter) and the time of the first frame of a video, the onset is
searched near the predicted arrivals.
"""

import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft, ndimage, signal

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

# km/s, for the predicted arrivals: apparent velocities (hypocentral distance / travel time) at 15-30 km in the
# Los Angeles area, from the catalog picks of the stations near the cameras (2026-11-20 M3.9; Vp/Vs about 1.8)
VP, VS = 5.7, 3.1
RANK = {'high': 0, 'normal': 1, 'low': 2}


# ---------------------------------------------------------------------------------------------- inputs

def read_cameras(path):
    """The list of cameras and their videos: camera_id, name, lat, lon, video, and optionally start_utc
    (ISO time of the first frame), view (what the camera looks at) and video_url (the video online, for the map)"""
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


def video_fps(video):
    """The frame rate of a video (30 if unknown)"""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate',
                          '-of', 'default=noprint_wrappers=1:nokey=1', video], capture_output=True, text=True).stdout
    try:
        num, den = out.strip().split('/')
        return float(num) / float(den) if float(den) else 30.0
    except ValueError:
        return 30.0


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


def _spectra(frames, win):
    f = frames.astype(np.float32)
    return fft.fft2((f - f.mean(axis=(1, 2), keepdims=True)) * win, axes=(1, 2), workers=-1)


def _correlation_peaks(X):
    """The shift (down, across) and the height of the peak of the phase correlation of each cross-power spectrum
    in X; sub-pixel from a parabola through the peak and its neighbors; the height is 1 for identical images and
    near 0 for unrelated ones"""
    c = fft.ifft2(X / (np.abs(X) + 1e-6), axes=(1, 2), workers=-1).real
    n, h, w = c.shape
    shifts, heights = np.empty((n, 2), np.float32), np.empty(n, np.float32)
    for k in range(n):
        iy, ix = np.unravel_index(np.argmax(c[k]), (h, w))
        heights[k] = c[k][iy, ix]
        sub = []
        for i, size, line in ((iy, h, c[k][:, ix]), (ix, w, c[k][iy])):
            cm, c0, cp = line[(i - 1) % size], line[i], line[(i + 1) % size]
            den = cm - 2 * c0 + cp
            frac = 0.5 * (cm - cp) / den if den < 0 else 0.0
            sub.append(((i + frac + size / 2) % size) - size / 2)
        shifts[k] = sub
    return shifts, heights


def _window(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def image_shifts(frames, ref, chunk=100, heights=False):
    """The shift (down, across) of each frame relative to frame ref, by phase correlation, in pixels of the
    frames (and the heights of the correlation peaks)"""
    n, h, w = frames.shape
    win = _window(h, w)
    R = np.conj(_spectra(frames[ref:ref + 1], win))
    shifts, peaks = np.empty((n, 2), np.float32), np.empty(n, np.float32)
    for a in range(0, n, chunk):
        shifts[a:a + chunk], peaks[a:a + chunk] = _correlation_peaks(R * _spectra(frames[a:a + chunk], win))
    return (shifts, peaks) if heights else shifts


def lagged_shifts(frames, lag, chunk=100):
    """The shift of each frame relative to the frame lag frames before it, and the height of the correlation
    peak (nan for the first lag frames)"""
    n, h, w = frames.shape
    win = _window(h, w)
    shifts, peaks = np.full((n, 2), np.nan, np.float32), np.full(n, np.nan, np.float32)
    for a in range(lag, n, chunk):
        b = min(a + chunk, n)
        F = _spectra(frames[a - lag:b], win)
        shifts[a:b], peaks[a:b] = _correlation_peaks(np.conj(F[:b - a]) * F[lag:])
    return shifts, peaks


def runs(mask):
    """The runs of True in mask, as [(start, end)] with end exclusive"""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


MOVE_PX_S = 1.0      # the image drifting faster than this (pixels of the 320-wide frames per s): the camera moves
JITTER_FACTOR = 8.0  # shaking: jitter above this many times the calm jitter of the camera
JITTER_FLOOR = 0.005  # px per frame


def analyze_motion(frames, fps, lo=0, hi=None, quiet_at=None, lead_s=0.5, calm_s=2.0):
    """The views and the shaking of a video. The view changes where consecutive frames do not match (a cut or a
    switch of preset), where frames 1 s apart do not match (a zoom), or where the image drifts steadily (the
    camera panning or tilting); between them are the stable views. The shaking is the jitter of the whole image
    back and forth within a stable view: the mean shift between consecutive frames, above JITTER_FACTOR times
    its median. The episode searched for starts in frames lo-hi; quiet_at is the frame for BEFORE when there is
    no shaking (e.g., at the predicted P arrival)."""
    n = len(frames)
    hi = n if hi is None else min(hi, n)
    one_s = max(2, int(round(fps)))
    s1, q1 = lagged_shifts(frames, 1)
    s_1s, q_1s = lagged_shifts(frames, one_s) if n > one_s else (np.full((n, 2), np.nan), np.full(n, np.nan))
    cut = np.nan_to_num(q1, nan=1.0) < 0.3 * np.nanmedian(q1) if n > 1 else np.zeros(n, bool)
    lost = np.nan_to_num(q_1s, nan=1.0) < 0.3 * np.nanmedian(q_1s) if n > one_s else np.zeros(n, bool)
    steps = np.where(cut[:, None], 0.0, np.nan_to_num(s1))
    step = np.hypot(*steps.T)
    jitter = ndimage.uniform_filter1d(ndimage.median_filter(step, 3, mode='nearest'), max(1, one_s // 2))
    # the resting position of the image: the running mean (3 s) of its position, which shaking back and forth
    # hardly moves, but a pan, a tilt or a move to another preset does; and the path of the image in the same
    # time: the camera moves when its resting position drifts, and steadily (a quarter of the path or more)
    rest = ndimage.uniform_filter1d(np.cumsum(steps, axis=0), 3 * one_s, axis=0, mode='nearest')
    half = one_s // 2
    drift = np.zeros(n, np.float32)
    if n > 2 * half:
        drift[half:n - half] = np.hypot(*(rest[2 * half:] - rest[:n - 2 * half]).T) * fps / (2 * half)
    path = ndimage.uniform_filter1d(step, 3 * one_s, mode='nearest') * fps
    moving = np.zeros(n, bool)
    for a, b in runs((drift > MOVE_PX_S) & (drift > 0.25 * path)):
        a, b = trim_move(steps, rest, a, b, max(2, one_s // 3))
        moving[a:b] = True
    pad = max(1, int(0.3 * fps))
    changing = ndimage.binary_dilation(moving | cut | lost, iterations=pad)
    # a change after which the image still matches the one before it, at the same place, is a sudden change of
    # the scene (lights out, smoke, a collapse, something passing close to the camera), not of the view
    q_typical = float(np.nanmedian(q_1s)) if np.isfinite(q_1s).any() else 0.5
    sudden = []
    for a, b in runs(changing):
        if a > 0 and b < n and not moving[a:b].any():
            s, q = image_shifts(frames[[a - 1, b]], 0, heights=True)
            if matches(q[1], s[1], q_typical, frames.shape[2], loose=False):
                changing[a:b] = False
                sudden.append((a + pad, max(a + pad + 1, b - pad)))
    segments = [(a, b) for a, b in runs(~changing) if b - a >= one_s]
    changes = runs(changing)
    stable = np.zeros(n, bool)
    for a, b in segments:
        stable[a:b] = True
    base = float(np.median(jitter[stable])) if stable.any() else float(np.median(jitter))
    thr = max(JITTER_FLOOR, JITTER_FACTOR * base)
    out = dict(jitter=jitter, drift=drift, q1=q1, q_1s=q_1s, changing=changing, segments=segments,
               changes=changes, sudden=sudden, threshold=thr, detected=False)

    # the episodes of shaking: runs above the threshold within a view, merged across gaps shorter than 2 s
    episodes = []
    for a, b in segments:
        for ra, rb in runs((jitter[a:b] > thr)):
            ra, rb = a + ra, a + rb
            if episodes and episodes[-1][2] == (a, b) and ra - episodes[-1][1] < 2 * fps:
                episodes[-1][1] = rb
            else:
                episodes.append([ra, rb, (a, b)])
    episodes = [e for e in episodes if e[1] - e[0] >= max(2, int(0.3 * fps)) and lo <= e[0] < hi]
    if episodes:
        onset, end, (a, b) = max(episodes, key=lambda e: float(np.sum(jitter[e[0]:e[1]] / thr - 1)))
        end -= 1
        peak = onset + int(np.argmax(jitter[onset:end + 1]))
        out.update(detected=True, onset=onset, end=end, peak=peak, segment=(a, b),
                   ref=max(a, onset - max(1, int(lead_s * fps))), calm_before=bool(onset - a > calm_s * fps),
                   at_view_change=bool(end >= b - 1 - int(0.5 * fps)))
    else:
        target = 0 if quiet_at is None else int(np.clip(quiet_at, 0, n - 1))
        seg = next(((a, b) for a, b in segments if b > target), segments[-1] if segments else (0, n))
        out.update(segment=seg, ref=int(np.clip(target, seg[0], seg[1] - 1)))
    return out


def trim_move(steps, rest, a, b, k):
    """The frames a-b where the camera moves, trimmed to those where the image steps along the move (k frames
    in a row at half the typical step or more): the running mean that finds a move spreads it in time"""
    disp = rest[min(b, len(rest) - 1)] - rest[a]
    norm = float(np.hypot(*disp))
    if norm == 0 or b - a < k:
        return a, b
    along = steps[a:b] @ (disp / norm)
    fwd = along[along > 0]
    good = along >= 0.5 * float(np.median(fwd)) if len(fwd) else np.zeros(b - a, bool)
    sustained = np.convolve(good, np.ones(k), 'valid') >= k                   # good[i:i+k] all True
    if not sustained.any():
        return a, b
    first, last = int(np.argmax(sustained)), len(sustained) - 1 - int(np.argmax(sustained[::-1]))
    return a + first, a + last + k


def matches(q, shift, q_typical, width, loose=True):
    """Whether two frames show the same view, from the height q of the peak of their phase correlation (about
    q_typical for frames 1 s apart in a view, 0.02-0.05 for unrelated views) and their shift: a good match
    within a tenth of the width (loose), or a weaker one at nearly the same place (the scene changed in part)"""
    d = float(np.hypot(*shift))
    return bool((loose and q >= 0.5 * q_typical and d <= 0.1 * width) or (q >= 0.1 * q_typical and d <= 4))


RESONANCE_CONCENTRATION = 0.5   # a resonant mount: at least this fraction of the motion near one frequency,
RESONANCE_DAMPING = 0.05        # damped by less than this fraction of critical,
RESONANCE_CYCLES = 5            # and ringing for this many cycles (e-folding time of the envelope x frequency);
LONG_RING_S = 10.0              # or ringing longer than this, at any spread of frequencies (a structure's modes)


def mount_response(shifts, fps, onset, max_step=32.0):
    """The response of the camera's mount, from the image shift (n x 2, px) in one view, the shaking starting
    at frame onset: the dominant frequency, the fraction of the motion near it, the ring-down (e-folding time of
    the envelope, s) and the damping it implies. A camera on a loose or tall mount (a pole, a mast, a bracket)
    rings at its own frequency, lightly damped, long after the ground calms down: its image motion shows the
    mount more than the ground. A broadband, quickly damped motion is closer to the ground shaking. None if the
    series is too short or the shifts are not reliable (a jump of more than max_step px between frames)"""
    n = len(shifts)
    if n < 4 * fps or onset < 2 or n - onset < 3 * fps or not np.isfinite(shifts).all():
        return None
    if np.abs(np.diff(shifts, axis=0)).max() > max_step:                # the correlation jumped
        return None
    x = signal.detrend(shifts - np.median(shifts[:onset], axis=0), axis=0)
    seg = x[onset:]
    fr, p = signal.welch(seg, fps, nperseg=min(len(seg), int(8 * fps)), axis=0)
    p = p.sum(axis=1)
    band = (fr >= 0.5) & (fr <= 0.4 * fps)                             # below 80 % of the Nyquist frequency
    if not band.any() or p[band].sum() <= 0:
        return None
    k = int(np.flatnonzero(band)[np.argmax(p[band])])
    f0 = float(fr[k])
    near = band & (np.abs(fr - f0) <= max(0.25, 0.1 * f0))
    concentration = float(p[near].sum() / p[band].sum())
    # the envelope of the motion near f0 (along its main direction), smoothed over 1 s; the ring-down from its
    # peak until it stays below twice its level before the shaking for 1 s (or the end of the view)
    u = np.linalg.svd(seg - seg.mean(0), full_matrices=False)[2][0]
    b_, a_ = signal.butter(2, [max(0.3, 0.7 * f0), min(0.45 * fps, 1.3 * f0)], btype='band', fs=fps)
    env = np.abs(signal.hilbert(signal.filtfilt(b_, a_, x @ u)))
    env = ndimage.uniform_filter1d(env, max(1, int(round(fps))))
    noise = float(np.median(env[:onset])) + 1e-6
    ip = onset + int(np.argmax(env[onset:]))
    quiet = [ra for ra, rb in runs(env[ip:] < 2 * noise) if rb - ra >= fps or ip + rb == len(env) and rb - ra >= fps / 2]
    below = quiet[:1]
    stop = ip + (int(below[0]) if below else len(env) - ip)
    ring_s = damping = None
    if stop - ip >= max(3, int(0.5 * fps)):
        slope = np.polyfit(np.arange(stop - ip) / fps, np.log(env[ip:stop]), 1)[0]
        if slope < 0:
            ring_s = float(-1 / slope)
            damping = float(-slope / (2 * np.pi * f0))
    resonant = bool(ring_s is not None and (ring_s >= LONG_RING_S or (
        concentration >= RESONANCE_CONCENTRATION and damping <= RESONANCE_DAMPING and ring_s * f0 >= RESONANCE_CYCLES)))
    return dict(frequency_hz=round(f0, 2), concentration=round(concentration, 2),
                ring_s=None if ring_s is None else round(ring_s, 1),
                damping=None if damping is None else round(damping, 3),
                ringing_to_end=bool(not len(below)), resonant=resonant)


KICK_FACTOR = 6.0               # a kick: a step between frames larger than this many times the typical step
KICK_MIN_PX = 0.2               # and than this (px of the full frame)
IMPULSIVE_KICKS_PER_S = 0.3     # impulsive: at least this many kicks per second,
IMPULSIVE_SKEWNESS = 1.0        # with a one-sided motion (|skewness| at least this)
IMPULSIVE_KURTOSIS = 8.0        # or heavy tails (kurtosis at least this; 3 for a Gaussian motion)


def linearity(x, fps):
    """Whether the image motion (n x 2, px of the full frame, high-passed; during the shaking, or ambient) looks
    like the response of a linear mount: a pole or a bracket driven by the ground (or the wind) moves smoothly
    and symmetrically (skewness about 0, kurtosis about 3 for a Gaussian, 1.5 for a sine), and its steps between
    frames stay within a few times their typical size. Free play (a loose part knocking against a stop),
    slip, or glitches of the video give kicks (a jump within one frame), a one-sided motion and heavy tails: the
    motion is then not a filtered version of the ground motion, and no linear correction (a calibration of the
    mount's frequency and damping) recovers the ground motion from it. Returns the skewness (of the axis with
    the larger |skewness|), the larger kurtosis, the kicks (separate runs of large steps) and impulsive"""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 * fps or not np.isfinite(x).all():
        return None
    xc = x - x.mean(0)
    sd = xc.std(0) + 1e-12
    skew = (xc ** 3).mean(0) / sd ** 3
    kurt = (xc ** 4).mean(0) / sd ** 4
    d = np.diff(x, axis=0)
    typical = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0) + 1e-9
    big = (np.abs(d) > np.maximum(KICK_FACTOR * typical, KICK_MIN_PX)).any(axis=1)
    kicks = len(runs(ndimage.binary_closing(big, np.ones(3, bool)) | big))       # steps one frame apart: one kick
    k = int(np.argmax(np.abs(skew)))
    per_s = kicks / (n / fps)
    impulsive = bool(per_s >= IMPULSIVE_KICKS_PER_S and (abs(skew[k]) >= IMPULSIVE_SKEWNESS
                                                          or kurt.max() >= IMPULSIVE_KURTOSIS))
    return dict(skewness=round(float(skew[k]), 2), kurtosis=round(float(kurt.max()), 1), kicks=kicks,
                kicks_per_s=round(per_s, 2), impulsive=impulsive)


def linearity_text(lin):
    """The linearity of the image motion in words ('' if it looks linear)"""
    if not lin or not lin.get('impulsive'):
        return ''
    return (f"The motion is impulsive, not a linear response of the mount: {lin['kicks']} kicks (jumps within a "
            f"frame), skewness {lin['skewness']}, kurtosis {lin['kurtosis']}: free play (a loose part knocking), "
            f"slip or video glitches; its spectrum and size do not follow the ground motion.")


def same_view(frames, ref, segments, after, q_typical):
    """The last frame after frame after in the same view as frame ref: the end of the last stable view whose
    frames match it; (frame, shift) or (None, None)"""
    for a, b in reversed(segments):
        k = b - 1
        if k <= after:
            break
        s, q = image_shifts(frames[[ref, k]], 0, heights=True)
        if matches(q[1], s[1], q_typical, frames.shape[2]):
            return k, s[1]
    return None, None


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

def tile(paths, labels, out, size=(640, 360), placements=None):
    """A 2 x 2 tile of up to four frames, each fitted into size and labeled; placements (a list), if given,
    receives where each frame is in the tile, [x0, y0, width, height] (None for a missing frame)"""
    canvas = Image.new('RGB', (2 * size[0], 2 * size[1]), 'black')
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for k, (path, label) in enumerate(zip(paths, labels)):
        if placements is not None:
            placements.append(None)
        if not path:
            continue
        with Image.open(path) as im:
            im = im.convert('RGB')
            im.thumbnail(size)
            x0 = (k % 2) * size[0] + (size[0] - im.width) // 2
            y0 = (k // 2) * size[1] + (size[1] - im.height) // 2
            canvas.paste(im, (x0, y0))
            if placements is not None:
                placements[-1] = [x0, y0, im.width, im.height]
        box = draw.textbbox((x0 + 8, y0 + 6), label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 6), label, fill=(255, 220, 0), font=font)
    canvas.save(out, quality=90)
    return out


def plot_motion(t, d, sh, shift_px, scale, marks, title, out):
    """The frame differences, the jitter and the drift of the image (the view changes shaded), and the image
    shift during the shaking"""
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True, constrained_layout=True)
    axes[0].plot(t[1:], d, color='#4a6fa5', lw=1)
    axes[0].set_ylabel('frame difference', fontsize=9)
    axes[0].set_title(title, fontsize=10, loc='left')
    axes[1].semilogy(t, np.maximum(sh['jitter'] * scale, 1e-4), lw=1, color='#1f77b4', label='jitter, px per frame')
    axes[1].semilogy(t, np.maximum(sh['drift'] * scale, 1e-4), lw=1, color='#ff7f0e', label='drift, px/s')
    axes[1].axhline(sh['threshold'] * scale, color='#d62728', lw=0.8, ls='--', label='threshold of the camera vibration')
    axes[1].set_ylim(bottom=max(1e-4, 0.1 * sh['threshold'] * scale))
    axes[1].legend(frameon=False, fontsize=7, ncol=3, loc='upper right')
    axes[1].set_ylabel('image motion', fontsize=9)
    if shift_px is not None:
        axes[2].plot(t, shift_px[:, 1], lw=1, label='across')
        axes[2].plot(t, shift_px[:, 0], lw=1, label='down')
        axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_ylabel('image shift, px', fontsize=9)
    axes[2].set_xlabel('time in the video, s', fontsize=9)
    dt_frame = t[1] - t[0] if len(t) > 1 else 0
    for ax in axes:
        for a, b in sh['changes']:
            ax.axvspan(t[a], t[b - 1] + dt_frame, color='0.85', lw=0,
                       label='view changes' if ax is axes[0] and a == sh['changes'][0][0] else None)
        for a, b in sh['sudden']:
            ax.axvspan(t[a], t[b - 1] + dt_frame, color='#f4b6b6', lw=0,
                       label='sudden changes' if ax is axes[0] and a == sh['sudden'][0][0] else None)
        for name, tm in marks.items():
            if tm is not None:
                ax.axvline(tm, color='#555555', lw=0.7, ls=':')
                if ax is axes[0]:
                    ax.annotate(name, (tm, 1), xycoords=('data', 'axes fraction'), xytext=(2, -10),
                                textcoords='offset points', fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    if sh['changes'] or sh['sudden']:
        axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig(out, dpi=100)
    plt.close(fig)


def motion_items(m):
    """The motion of a camera in short statements, for the map and the console"""
    flags = set(m.get('flags', []))
    if 'no_video' in flags:
        return ['No video']
    if any(f.startswith('error') for f in flags):
        return ['Video could not be processed']
    items = []
    if m.get('shaking_detected'):
        items.append(f"Camera vibration {m['onset_s']:.1f}-{m['end_s']:.1f} s ({m['strong_duration_s']:.0f} s), "
                     f"up to {m['shake_amplitude_px']:g} px")
    elif 'shaking_detected' in m:
        items.append('No camera vibration detected')
    if mount := m.get('mount'):
        if mount['resonant']:
            more = '+' if mount['ringing_to_end'] else ''
            items.append(f"Mount resonance: {mount['frequency_hz']:g} Hz, rings {mount['ring_s']:g}{more} s "
                         f"(motion overstates the shaking)")
        else:
            items.append('No mount resonance: motion closer to the ground shaking')
    if 'shaking_until_a_view_change' in flags:
        items.append('Camera vibration lasts until the view changes')
    elif 'shaking_until_the_end_of_the_clip' in flags:
        items.append('Camera vibration lasts until the end of the video')
    if m.get('view_changes_s'):
        n = len(m['view_changes_s'])
        items.append(f"Camera pans or switches views {n} time{'s' if n > 1 else ''} (not the earthquake)")
    for a, _ in (m.get('sudden_changes_s') or [])[:3]:
        items.append(f'Sudden change of the scene at {a:g} s')
    if 'camera_moved' in flags:
        items.append(f"Camera shifted {m.get('permanent_shift_px')} px by the shaking")
    if 'scene_changed' in flags:
        pct = 100 * max(m.get('changed_fraction_after', 0), m.get('changed_fraction_latest', 0))
        items.append(f'Scene changed: {pct:.0f} % of the image')
    if 'latest_in_another_view' in flags:
        items.append('Latest frame shows another view')
    if 'dark_at_the_end' in flags:
        items.append('Dark at the end (possible outage)')
    if 'frozen_at_the_end' in flags:
        items.append('Frozen at the end (possible outage)')
    if 'predicted_arrivals_outside_the_video' in flags:
        items.append('Predicted arrivals outside the video')
    if 'too_short' in flags:
        items.append('Video too short')
    return items


def mount_text(mount):
    """The response of the mount in words"""
    ring = (f"ringing {mount['ring_s']} s{' or more' if mount['ringing_to_end'] else ''}"
            if mount.get('ring_s') is not None else 'no clear ring-down')
    damping = f", damping about {100 * mount['damping']:.1f} %" if mount.get('damping') is not None else ''
    near = f"{mount['frequency_hz']} Hz ({100 * mount['concentration']:.0f} % of the motion near it)"
    if mount['resonant']:
        return (f"The camera's mount resonates: dominant {near}, {ring}{damping}. The image motion shows the mount "
                f"(pole, mast or building) more than the ground shaking; do not take its size as the intensity.")
    return (f"The image motion is not a mount resonance: dominant {near}, {ring}{damping}; it may be closer to "
            f"the ground shaking (a stiff mount), or come from a loose part or the video (one-sided jumps); its "
            f"size depends on the mount and the zoom.")


def motion_text(m):
    """The motion of a camera in words, for the review"""
    if 'no_video' in m['flags']:
        return 'No video for this camera.'
    parts = []
    changes = m.get('view_changes_s') or []
    if changes:
        listed = ', '.join(f'{a}-{b} s' for a, b in changes[:8]) + (', ...' if len(changes) > 8 else '')
        parts.append(f'The view changes {len(changes)} times (the camera pans, zooms or switches views; this is '
                     f'not the earthquake): {listed}.')
    sudden = m.get('sudden_changes_s') or []
    if sudden:
        parts.append('The scene changes suddenly in the same view (look for what changed) at '
                     + ', '.join(f'{a}-{b} s' for a, b in sudden[:8]) + '.')
    if m.get('shaking_detected'):
        parts.append(f"The camera vibrates (the whole image jitters within one view) from {m['onset_s']} s to "
                     f"{m['end_s']} s in the video ({m['strong_duration_s']} s), peak at {m['peak_s']} s; image motion about "
                     f"{m['shake_amplitude_px']} px, peak image velocity {m['peak_image_velocity_px_s']} px/s, "
                     f"permanent shift {m['permanent_shift_px']} px.")
    elif 'shaking_detected' in m:
        parts.append('No camera vibration detected in the video.')
    if mount := m.get('mount'):
        parts.append(mount_text(mount))
    for key in ('changed_fraction_after', 'changed_fraction_latest'):
        if key in m:
            parts.append(f"Image changed between BEFORE and {key.rsplit('_', 1)[1].upper()}: "
                         f"{100 * m[key]:.1f} % of the area.")
    if 'latest_in_another_view' in m['flags']:
        parts.append('LATEST shows another view than BEFORE: do not compare them for changes.')
    elif changes and 'latest' in m.get('frames', {}):
        parts.append(f"LATEST is the last frame in the same view as BEFORE ({m['frames']['latest']['video_time_s']} s).")
    flags = [f for f in m['flags'] if not f.startswith('view_changes')]
    if flags:
        parts.append('Flags: ' + ', '.join(flags) + '.')
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
    if cam.get('video_url'):
        m['video_url'] = cam['video_url']
    if cam.get('start_utc'):
        m['start_utc'] = cam['start_utc']
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
        m['predicted_p_s'], m['predicted_s_s'] = round(origin + hypo / VP, 2), round(origin + hypo / VS, 2)
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
    lo = int((lo_s - start) * motion_fps) if lo_s else 0
    hi = int((hi_s - start) * motion_fps) if hi_s is not None else n
    quiet_at = int((m['predicted_p_s'] - 2 - start) * motion_fps) if 'predicted_p_s' in m else None
    sh = analyze_motion(frames, motion_fps, max(lo, 0), max(hi, 1), quiet_at)
    scale = width / motion_width
    m['shaking_detected'] = sh['detected']
    m['jitter_threshold_px'] = round(sh['threshold'] * scale, 4)
    if sh['changes']:
        m['view_changes_s'] = [[round(float(t[a]), 1), round(float(t[b - 1]), 1)] for a, b in sh['changes']]
        m['flags'].append(f"view_changes ({len(sh['changes'])})")
    if sh['sudden']:
        m['sudden_changes_s'] = [[round(float(t[a]), 1), round(float(t[b - 1]), 1)] for a, b in sh['sudden']]

    brightness = frames.reshape(n, -1).mean(axis=1)
    tail = max(1, int(2 * motion_fps))
    shift_px = None
    a, b = sh['segment']
    ref = sh['ref']
    picks = dict(before=ref, peak=None, after=None, latest=None)
    if sh['detected']:
        onset, peak, end = sh['onset'], sh['peak'], sh['end']
        shift_px = np.full((n, 2), np.nan, np.float32)
        shift_px[a:b] = image_shifts(frames[a:b], ref - a) * scale        # within the view of the shaking
        before = shift_px[max(a, ref - tail):ref + 1]
        during = shift_px[onset:end + 1]
        after_i = min(b - 1, end + int(motion_fps))
        settled = shift_px[after_i:min(b, after_i + tail)] if after_i > end else shift_px[:0]
        vel = np.linalg.norm(np.diff(ndimage.uniform_filter1d(shift_px[a:b], 3, axis=0), axis=0), axis=1)
        permanent = float(np.linalg.norm(np.median(settled, 0) - np.median(before, 0))) if len(settled) else None
        m.update(onset_s=round(float(t[onset]), 2), peak_s=round(float(t[peak]), 2), end_s=round(float(t[end]), 2),
                 strong_duration_s=round((end - onset) / motion_fps, 2),
                 shake_amplitude_px=round(float(np.max(np.ptp(during, axis=0)) / 2), 1),
                 peak_image_velocity_px_s=round(float(vel[max(onset - a - 1, 0):end - a].max()) * motion_fps, 1),
                 permanent_shift_px=None if permanent is None else round(permanent, 1),
                 calm_frames_before=sh['calm_before'])
        if not sh['calm_before']:
            m['flags'].append('no_calm_frames_before_the_shaking')
        if sh['at_view_change']:
            m['flags'].append('shaking_until_a_view_change' if b < n else 'shaking_until_the_end_of_the_clip')
        if permanent and permanent > 3:
            m['flags'].append('camera_moved')
        picks.update(peak=peak, after=after_i)
        # the mount: the window of the shaking and its ring-down, in its view, at the frame rate of the video
        fps_v = min(30.0, video_fps(video))
        ta, tb = max(float(t[a]), float(t[onset]) - 2), min(float(t[b - 1]), float(t[end]) + 15)
        fr = decode_small(video, motion_width, h_small, fps_v, ta, tb - ta)
        if len(fr) > 4 * fps_v:
            mount = mount_response(image_shifts(fr, 0), fps_v, int(round((float(t[onset]) - ta) * fps_v)),
                                   max_step=0.1 * motion_width)
            if mount:
                m['mount'] = mount
                if mount['resonant']:
                    m['flags'].append('resonant_mount')

    # the latest frame in the same view as BEFORE (cameras that tour their views come back to it)
    last = picks['after'] if picks['after'] is not None else ref
    k, latest_shift = None, None
    if not any(c0 > last for c0, _ in sh['changes']):           # the view does not change after AFTER
        if n - 1 > last:
            k = n - 1
            s, _ = image_shifts(frames[[ref, k]], 0, heights=True)
            latest_shift = s[1] if np.hypot(*s[1]) <= 0.1 * motion_width else np.zeros(2)
    else:
        q_typical = float(np.nanmedian(sh['q_1s'])) if np.isfinite(sh['q_1s']).any() else 0.5
        k, latest_shift = same_view(frames, ref, sh['segments'], last, q_typical)
        if k is None and n - 1 > last:
            k = n - 1
            m['flags'].append('latest_in_another_view')
    picks['latest'] = k

    # the end of the video: dark or frozen
    if brightness[-tail:].mean() < max(20.0, 0.3 * brightness[ref]):
        m['flags'].append('dark_at_the_end')
    if n > 2 * tail and (d[-tail:] < 0.05).all() and not (d[:tail] < 0.05).all():
        m['flags'].append('frozen_at_the_end')

    # the change between before and after, and before and the latest frame, in the same view
    if picks['after'] is not None and picks['after'] != ref:
        m['changed_fraction_after'] = round(changed_fraction(frames[ref], frames[picks['after']],
                                                             shift_px[picks['after']] / scale), 4)
    if picks['latest'] is not None and latest_shift is not None:
        m['changed_fraction_latest'] = round(changed_fraction(frames[ref], frames[picks['latest']], latest_shift), 4)
    if max(m.get('changed_fraction_after', 0), m.get('changed_fraction_latest', 0)) > 0.03:
        m['flags'].append('scene_changed')

    # the frames at full resolution, and the tile
    other = ' (OTHER VIEW)' if 'latest_in_another_view' in m['flags'] else ''
    if picks['after'] is not None and (picks['latest'] is None or picks['latest'] - picks['after'] < 2 * motion_fps):
        mid = (sh['peak'] + sh['end']) // 2
        picks['latest'] = None if mid == sh['peak'] else mid
        order = [('before', 'BEFORE'), ('peak', 'PEAK SHAKING'), ('latest', 'LATE SHAKING'), ('after', 'AFTER')]
    else:
        order = [('before', 'BEFORE'), ('peak', 'PEAK SHAKING'), ('after', 'AFTER'), ('latest', 'LATEST' + other)]
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
    placed = []
    m['tile'] = os.path.relpath(tile(paths, labels, os.path.join(cdir, 'tile.jpg'), placements=placed), outdir)
    m['tile_layout'] = {key: p for (key, _), p in zip(order, placed) if p is not None}
    shutil.rmtree(os.path.join(cdir, 'original'), ignore_errors=True)      # the frames of a former masking
    marks = {key: float(t[picks[key]]) if picks.get(key) is not None else None for key, _ in order}
    plot_motion(t, d, sh, shift_px, scale, marks, f'{cid}: {m["name"]}', os.path.join(cdir, 'motion.png'))
    m['motion_plot'] = os.path.relpath(os.path.join(cdir, 'motion.png'), outdir)
    np.savez_compressed(os.path.join(cdir, 'motion.npz'), t=t, d=d, brightness=brightness,
                        jitter_px=sh['jitter'] * scale, drift_px_s=sh['drift'] * scale, changing=sh['changing'],
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
        m['motion_items'] = motion_items(m)
        metrics.append(m)
        print(f"{m['camera_id']:>20s}  {m['priority']:6s}  {'; '.join(m['motion_items'])}")
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    keys = ('camera_id', 'name', 'view', 'distance_km', 'priority', 'tile', 'frames', 'motion_plot', 'motion_text')
    tasks = [{k: m.get(k) for k in keys} for m in sorted(metrics, key=lambda m: RANK[m['priority']])]
    with open(os.path.join(out, 'tasks.json'), 'w') as f:
        json.dump(tasks, f, indent=1)
    print(f'{len(tasks)} cameras -> {out}/tasks.json')
    return tasks
