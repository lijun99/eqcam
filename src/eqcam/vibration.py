"""vibration: the vibration of each camera and of the objects in its view, for scientific use (optional:
eqcam report --vibration)

For each camera with shaking, the shaking and its ring-down, in the view of the shaking, are decoded again at
the frame rate of the video, 640 px wide. A grid of patches is tracked by phase correlation; per frame, an affine
transform (shift, roll, scale and shear about the image center; the shear takes the rolling shutter of a camera
vibrating fast) is fitted to the patches (least squares, then again without the patches that do not follow it):
the motion of the camera. What is left in a patch is the motion of what it shows relative to the camera. A patch
that oscillates during the shaking well above its level before, at a frequency of its own (a spectral peak, not
coherent with the camera's motion), tracked reliably (no jumps, no loss of the match: not traffic or people
passing), is a vibrating object (a pole, a sign, a tree, wires, a structure); a patch whose residual mirrors the
camera's motion does not move with the scene (burned-in text, a logo) and is set aside.

For the camera and the objects: the displacement as a function of time, the spectrum (the frequency
distribution) and, for the camera, a spectrogram, in cameras/<id>/vibration.png (drawn on the frame saved by
prepare, with the people blurred after eqcam report), vibration.json and vibration.npz; for the event,
vibration.html (a map of the cameras by their vibration) and vibration_summary.png (the dominant frequencies
against the distance, and the spectra of the cameras).

Displacements are in pixels of the full frame: their size depends on the zoom, the distance of the object and
the mount (see eqcam.frames.mount_response), so compare frequencies and shapes across cameras, not amplitudes.
"""

import html
import json
import math
import os

import matplotlib
import numpy as np
from PIL import Image
from scipy import signal

from .frames import decode_small, image_shifts, video_fps
from .report import BASEMAPS_JS, FIT_JS, PANEL_CSS, PANEL_JS, title_of

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402

# the reference palette: categorical slots in a fixed order, ink, grid, surface
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, MUTED = '#0b0b0b', '#52514e', '#e4e3df', '#8a8f98'
CLASS_COLORS = {'resonant mount': SERIES[1], 'no mount resonance': SERIES[0], 'not analyzed': MUTED}

WIDTH = 640                 # px, the width of the frames tracked
CELLS = (10, 6)             # the grid of patches, across and down
MIN_HEIGHT = 0.15           # a patch with texture: median height of its correlation peak
OBJECT_FACTOR = 4.0         # an object: its oscillation during the shaking, over its level before,
OBJECT_MIN_PX = 0.03        # at least this (px of the tracked frames),
MIN_OSCILLATION = 0.6       # and mostly oscillation (fraction of its variance above 0.5 Hz),
MIN_CONCENTRATION = 0.25    # at a frequency of its own (fraction of its power near its peak),
MAX_CAMERA_COHERENCE = 0.5  # not coherent with the camera's motion at its frequency (rolling shutter, lens),
MAX_BAD_FRAMES = 0.1        # tracked reliably in all but this fraction of the frames
MAX_STEP_PX = 2.0           # (a step larger than this between frames, px of the tracked frames, is a jump)
MAX_OBJECTS = 4             # objects plotted (all are in vibration.json)


def window_of(m):
    """The part of the video analyzed, (start, end) s: 5 s before the onset to 15 s after the end of the shaking,
    within the view of the shaking"""
    on, end = m['onset_s'], m['end_s']
    lo, hi = 0.0, m.get('duration_s') or end + 15
    for a, b in sorted(m.get('view_changes_s') or []):
        if b <= on:
            lo = max(lo, b)
        elif a >= on:
            hi = min(hi, a)
            break
    return max(lo, on - 5), min(hi, end + 15)


def highpass(x, fps, f=0.5):
    b, a = signal.butter(2, f, btype='high', fs=fps)
    return signal.filtfilt(b, a, x, axis=0) if len(x) > 15 else x - x.mean(0)


def spectrum(x, fps):
    """The power spectrum of a motion (n x 2, summed over both axes) and its dominant frequency, in 0.5 Hz to 80 %
    of the Nyquist frequency, with the fraction of the power near it"""
    fr, p = signal.welch(x, fps, nperseg=min(len(x), int(8 * fps)), axis=0)
    p = p.sum(axis=1)
    band = (fr >= 0.5) & (fr <= 0.4 * fps)
    if not band.any() or p[band].sum() <= 0:
        return fr, p, None, 0.0
    k = int(np.flatnonzero(band)[np.argmax(p[band])])
    near = band & (np.abs(fr - fr[k]) <= max(0.25, 0.1 * fr[k]))
    return fr, p, float(fr[k]), float(p[near].sum() / p[band].sum())


def coherence_at(x, y, fps, f):
    """The largest magnitude-squared coherence between the axes of two motions (n x 2) near frequency f: a
    residual that follows the camera (in phase or not: a rolling shutter goes with its velocity) is close to 1"""
    if f is None or len(x) < 16:
        return 0.0
    best = 0.0
    for i in range(2):
        for j in range(2):
            if np.std(x[:, i]) > 0 and np.std(y[:, j]) > 0:
                fr, c = signal.coherence(x[:, i], y[:, j], fps, nperseg=min(len(x), int(4 * fps)))
                near = np.abs(fr - f) <= max(0.25, 0.1 * f)
                if near.any():
                    best = max(best, float(c[near].max()))
    return best


def fit_camera(local, centers, use):
    """The motion of the camera per frame, an affine transform about the image center (shift across, shift down,
    and the four gradients), fitted to the shifts of the patches used (local: n x P x 2, down and across); and
    the shifts it predicts. Roll = (d down/dx - d across/dy) / 2, scale = (d across/dx + d down/dy) / 2"""
    x, y = centers[:, 0], centers[:, 1]
    P = len(x)
    A = np.zeros((2 * P, 6))
    A[:P, 0], A[:P, 2], A[:P, 3] = 1, x, y                   # across = p + c1 x + c2 y
    A[P:, 1], A[P:, 4], A[P:, 5] = 1, x, y                   # down = q + c3 x + c4 y
    rows = np.concatenate([use, use])
    b = np.concatenate([local[:, :, 1], local[:, :, 0]], axis=1)          # n x 2P
    params = np.linalg.lstsq(A[rows], b[:, rows].T, rcond=None)[0].T        # n x 6
    pred = params @ A.T
    return params, np.stack([pred[:, P:], pred[:, :P]], axis=2)            # predicted (down, across)


def analyze_camera(out, m, fps_max=30.0):
    """The vibration of one camera (with shaking) and of the objects in its view; writes vibration.png, .json,
    .npz in its folder and returns the summary (None if it cannot be analyzed)"""
    if not m.get('shaking_detected') or not m.get('video') or not os.path.exists(m['video']):
        return None
    ta, tb = window_of(m)
    fps = min(fps_max, video_fps(m['video']))
    width, height = m['width'], m['height']
    h_small = max(2, int(round(height * WIDTH / width / 2)) * 2)
    frames = decode_small(m['video'], WIDTH, h_small, fps, ta, tb - ta)
    n = len(frames)
    onset = int(round((m['onset_s'] - ta) * fps))
    if n < 4 * fps or onset < 2:
        return None
    scale = width / WIDTH
    t = ta + np.arange(n) / fps

    # the patches, tracked relative to the first frame (before the shaking)
    nx, ny = CELLS
    cw, ch = WIDTH // nx, h_small // ny
    boxes = [(i * cw, j * ch, (i + 1) * cw, (j + 1) * ch) for j in range(ny) for i in range(nx)]
    local = np.empty((n, len(boxes), 2), np.float32)
    height_q = np.empty((n, len(boxes)), np.float32)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        local[:, k], height_q[:, k] = image_shifts(frames[:, y0:y1, x0:x1], 0, heights=True)
    centers = np.array([((x0 + x1) / 2 - WIDTH / 2, (y0 + y1) / 2 - h_small / 2) for x0, y0, x1, y1 in boxes])
    textured = np.median(height_q, axis=0) >= MIN_HEIGHT
    reliable = (height_q >= 0.5 * np.median(height_q, axis=0)).mean(axis=0) >= 0.8
    usable = textured & reliable
    if usable.sum() < 6:
        return None

    # the camera: fitted to the usable patches, then again without those that do not follow it
    params, pred = fit_camera(local, centers, usable)
    resid = local - pred
    rms = np.sqrt(np.mean(highpass(resid[onset:], fps) ** 2, axis=(0, 2)))
    follows = usable & (rms <= 3 * np.median(rms[usable]) + 1e-4)
    if follows.sum() >= 6:
        params, pred = fit_camera(local, centers, follows)
        resid = local - pred
    cam = params[:, :2] * scale                               # px of the full frame, (across, down)
    cam_hp = highpass(cam - cam[:onset].mean(0), fps)
    fr, cam_psd, cam_f0, cam_conc = spectrum(cam_hp[onset:], fps)

    # the objects: oscillating on their own during the shaking; the overlays mirror the camera
    cam_motion = highpass(local - local[:onset].mean(0), fps)[onset:]       # the motion of each patch itself
    cam_640 = highpass(params[:, :2] - params[:onset, :2].mean(0), fps)[onset:]
    step = np.hypot(*np.diff(local, axis=0, prepend=local[:1]).transpose(2, 0, 1))
    bad = (height_q < 0.6 * np.median(height_q, axis=0)) | (step > MAX_STEP_PX)
    robust = lambda a: float(1.4826 * np.median(np.abs(a - np.median(a, axis=0)))) + 1e-5  # noqa: E731
    objects, overlays, set_aside = [], 0, 0
    rh = highpass(resid - resid[:onset].mean(0), fps)
    for k in np.flatnonzero(usable):
        during, before = rh[onset:, k], rh[:onset, k]
        amp, noise = robust(during), robust(before)
        own = robust(cam_motion[:, k])
        if own < 0.3 * amp:                                   # it does not move while its residual does
            overlays += 1
            continue
        if bad[:, k].mean() > MAX_BAD_FRAMES or np.abs(during).max() > 0.08 * cw:   # traffic, people, jumps
            set_aside += 1
            continue
        raw = resid[onset:, k] - resid[onset:, k].mean(0)
        osc = float((during ** 2).sum() / max((raw ** 2).sum(), 1e-12))
        if amp >= OBJECT_FACTOR * noise and amp >= OBJECT_MIN_PX and osc >= MIN_OSCILLATION:
            _, p_k, f_k, c_k = spectrum(during, fps)
            corr = coherence_at(during, cam_640, fps, f_k)
            if corr > MAX_CAMERA_COHERENCE or c_k < MIN_CONCENTRATION:
                set_aside += 1
                continue
            x0, y0, x1, y1 = boxes[k]
            objects.append(dict(patch=int(k), box=[round(v * scale) for v in (x0, y0, x1, y1)],
                                frequency_hz=None if f_k is None else round(f_k, 2), concentration=round(c_k, 2),
                                rms_px=round(amp * scale, 3), peak_px=round(float(np.abs(during).max()) * scale, 3),
                                above_before=round(amp / noise, 1), camera_coherence=round(corr, 2), psd=p_k))
    objects.sort(key=lambda o: -o['rms_px'])
    for i, o in enumerate(objects, 1):
        o['id'] = i

    roll, zoom = (params[:, 4] - params[:, 3]) / 2, (params[:, 2] + params[:, 5]) / 2
    summary = dict(window_s=[round(ta, 2), round(tb, 2)], fps=round(fps, 2), px_scale=round(scale, 3),
                   camera=dict(frequency_hz=None if cam_f0 is None else round(cam_f0, 2),
                               concentration=round(cam_conc, 2),
                               peak_px=round(float(np.abs(cam_hp[onset:]).max()), 3),
                               rms_px=round(float(np.sqrt(np.mean(cam_hp[onset:] ** 2))), 3),
                               roll_rms_mrad=round(float(np.std(roll[onset:])) * 1000, 3),
                               scale_rms_ppm=round(float(np.std(zoom[onset:])) * 1e6, 1)),
                   mount=m.get('mount'), patches_used=int(usable.sum()), overlays=overlays, set_aside=set_aside,
                   objects=[{k: v for k, v in o.items() if k != 'psd'} for o in objects])
    cdir = os.path.join(out, 'cameras', m['camera_id'])
    with open(os.path.join(cdir, 'vibration.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    sel = objects[:MAX_OBJECTS]
    obj_series = np.stack([rh[:, o['patch']] * scale for o in sel], 1) if sel else np.zeros((n, 0, 2))
    np.savez_compressed(os.path.join(cdir, 'vibration.npz'), t=t, camera_px=cam, roll=roll, scale=zoom,
                        affine=params, freq=fr, camera_psd=cam_psd, object_px=obj_series,
                        object_ids=[o['id'] for o in sel], object_psd=np.array([o['psd'] for o in sel]))
    plot_camera(out, m, summary, t, cam_hp, fr, cam_psd, sel, obj_series, fps, onset, scale,
                os.path.join(cdir, 'vibration.png'))
    summary['plot'] = os.path.relpath(os.path.join(cdir, 'vibration.png'), out)
    summary['camera_psd'] = cam_psd.tolist()
    summary['freq'] = fr.tolist()
    return summary


def main_direction(x):
    """The motion (n x 2) along its main direction"""
    u = np.linalg.svd(x - x.mean(0), full_matrices=False)[2][0]
    return x @ u


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(labelsize=12, colors=INK2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def plot_camera(out, m, summary, t, cam_hp, fr, cam_psd, objects, obj_series, fps, onset, scale, path):
    fig = plt.figure(figsize=(13, 13), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    on, end = m['onset_s'], m['end_s']
    title = f"{m['camera_id']} {m.get('name', '')}, {m.get('distance_km', '?')} km"
    fig.suptitle(f'{title}: vibration of the camera and of the objects in its view\n'
                 'px: pixels of the full frame; their size depends on the zoom, the distance and the mount:\n'
                 'compare frequencies and shapes across cameras, not amplitudes', fontsize=15, color=INK, x=0.01,
                 ha='left')

    # the frame with the objects
    ax = fig.add_subplot(gs[0, 0])
    before = (m.get('frames') or {}).get('before')
    if before and os.path.exists(os.path.join(out, before['file'])):
        with Image.open(os.path.join(out, before['file'])) as im:
            ax.imshow(im.convert('RGB'), extent=(0, m['width'], m['height'], 0))
    for o in objects:
        x0, y0, x1, y1 = o['box']
        color = SERIES[(o['id'] - 1) % len(SERIES)]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2))
        ax.text(x0 + 4, y0 + 4, str(o['id']), color=INK, fontsize=13, va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, lw=1))
    ax.set_xlim(0, m['width'])
    ax.set_ylim(m['height'], 0)
    ax.axis('off')
    n_obj = len(summary['objects'])
    ax.set_title(f'{n_obj} object{"s" if n_obj != 1 else ""} vibrating on {"their" if n_obj != 1 else "its"} own'
                 + (f' (the {len(objects)} strongest outlined)' if n_obj > len(objects) else '')
                 + (f'\nset aside: {summary["overlays"]} overlay, {summary["set_aside"]} other patches'
                    if summary['overlays'] or summary['set_aside'] else ''),
                 fontsize=13, color=INK2, loc='left')

    # the camera: displacement and spectrum
    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(on, end, color=GRID, alpha=0.6, lw=0)
    ax.plot(t, cam_hp[:, 0], color=SERIES[0], lw=1, label='across')
    ax.plot(t, cam_hp[:, 1], color=SERIES[1], lw=1, label='down')
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.set_ylabel('camera displacement, px', fontsize=13, color=INK2)
    ax.set_xlabel('time in the video, s (shaded: the shaking)', fontsize=13, color=INK2)
    style(ax)
    ax = fig.add_subplot(gs[1, 0])
    band = fr > 0
    ax.semilogy(fr[band], np.maximum(cam_psd[band], 1e-12), color=SERIES[0], lw=1.2)
    cam = summary['camera']
    mount = summary.get('mount') or {}
    if cam['frequency_hz']:
        ax.axvline(cam['frequency_hz'], color=INK2, lw=0.8, ls=':')
        label = f"{cam['frequency_hz']:g} Hz, {100 * cam['concentration']:.0f} % of the power near it"
        if mount.get('resonant'):
            label += f"\nresonant mount, rings {mount.get('ring_s')} s"
        right = cam['frequency_hz'] < 0.5 * fr[band].max()          # the label on the side away from the peak
        ax.text(0.98 if right else 0.02, 0.97, label, transform=ax.transAxes, ha='right' if right else 'left',
                va='top', fontsize=12, color=INK, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax.set_xlabel('frequency, Hz', fontsize=13, color=INK2)
    ax.set_ylabel('camera power, px² / Hz', fontsize=13, color=INK2)
    style(ax)
    ax = fig.add_subplot(gs[1, 1])
    x = main_direction(cam_hp)
    nper = max(16, int(2 * fps))
    if len(x) > nper:
        f_s, t_s, sxx = signal.spectrogram(x, fps, nperseg=nper, noverlap=int(0.85 * nper))
        db = 10 * np.log10(np.maximum(sxx, 1e-12))
        ax.pcolormesh(t[0] + t_s, f_s, db, cmap='Blues', vmin=np.percentile(db, 50), vmax=db.max(), shading='auto')
    ax.set_ylabel('frequency, Hz', fontsize=13, color=INK2)
    ax.set_xlabel('time in the video, s', fontsize=13, color=INK2)
    ax.set_title('camera: spectrogram (darker: more power)', fontsize=13, color=INK2, loc='left')
    style(ax)

    # the objects: displacement relative to the camera, and spectra
    ax = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    if objects:
        ax.axvspan(on, end, color=GRID, alpha=0.6, lw=0)
        for i, o in enumerate(objects):
            color = SERIES[(o['id'] - 1) % len(SERIES)]
            ax.plot(t, main_direction(obj_series[:, i]), color=color, lw=1, label=f"object {o['id']}")
            p = np.asarray(o['psd'])
            ax2.semilogy(fr[band], np.maximum(p[band] / max(p[band].max(), 1e-12), 1e-6), color=color, lw=1.2,
                         label=f"object {o['id']}: {o['frequency_hz']} Hz")
        ax.legend(frameon=False, fontsize=12, loc='upper right')
        ax2.legend(frameon=False, fontsize=12, loc='upper right')
    else:
        for a in (ax, ax2):
            a.text(0.5, 0.5, 'no object vibrating on its own\nabove its level before the shaking',
                   transform=a.transAxes, ha='center', va='center', fontsize=13, color=INK2)
    if objects:
        ax.set_ylabel('object, relative to the camera, px', fontsize=13, color=INK2)
        ax.set_xlabel('time in the video, s', fontsize=13, color=INK2)
        ax2.set_xlabel('frequency, Hz', fontsize=13, color=INK2)
        ax2.set_ylabel('object power, normalized', fontsize=13, color=INK2)
        style(ax)
        style(ax2)
    else:
        ax.axis('off')
        ax2.axis('off')
    fig.savefig(path, dpi=100, facecolor='white')
    plt.close(fig)


def class_of(r):
    v = r.get('vibration')
    if not v:
        return 'not analyzed'
    return 'resonant mount' if (v.get('mount') or {}).get('resonant') else 'no mount resonance'


def summary_png(rows, event, out):
    """The dominant frequencies of the cameras and their objects against the distance, and the spectra of the
    cameras (normalized)"""
    done = [r for r in rows if r.get('vibration')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for r in done:
        v, d = r['vibration'], r.get('distance_km')
        if d is None:
            continue
        if v['camera']['frequency_hz']:
            a1.scatter(d, v['camera']['frequency_hz'], s=60, color=CLASS_COLORS[class_of(r)], ec='white', lw=1.5,
                       zorder=3)
            a1.annotate(r['camera_id'], (d, v['camera']['frequency_hz']), xytext=(6, 4), textcoords='offset points',
                        fontsize=12, color=INK)
        for o in v['objects']:
            if o['frequency_hz']:
                a1.scatter(d, o['frequency_hz'], s=22, marker='D', color=SERIES[2], ec='white', lw=1, zorder=2)
    for name in ('resonant mount', 'no mount resonance'):
        a1.scatter([], [], s=60, color=CLASS_COLORS[name], label=f'camera: {name}')
    if any(r['vibration']['objects'] for r in done):
        a1.scatter([], [], s=22, marker='D', color=SERIES[2], label='object in a view')
    a1.legend(frameon=False, fontsize=12, loc='upper right')
    a1.set_xlabel('distance from the epicenter, km', fontsize=13, color=INK2)
    a1.set_ylabel('dominant frequency, Hz', fontsize=13, color=INK2)
    a1.set_title('dominant frequency of the cameras and objects', fontsize=13, color=INK2, loc='left')
    style(a1)
    for i, r in enumerate(sorted(done, key=lambda r: r.get('distance_km') or 0)):
        v = r['vibration']
        fr, p = np.asarray(v['freq']), np.asarray(v['camera_psd'])
        band = fr > 0
        color = SERIES[i] if i < len(SERIES) else MUTED
        a2.semilogy(fr[band], np.maximum(p[band] / max(p[band].max(), 1e-12), 1e-6), color=color, lw=1.2,
                    label=f"{r['camera_id']} ({r.get('distance_km')} km)" if i < len(SERIES) else None)
    a2.legend(frameon=False, fontsize=12, loc='upper right')
    a2.set_xlabel('frequency, Hz', fontsize=13, color=INK2)
    a2.set_ylabel('camera power, normalized', fontsize=13, color=INK2)
    a2.set_title('spectra of the camera motion', fontsize=13, color=INK2, loc='left')
    style(a2)
    fig.suptitle(f'{title_of(event)}: vibration', fontsize=14, color=INK, x=0.01, ha='left')
    fig.savefig(os.path.join(out, 'vibration_summary.png'), dpi=100, facecolor='white')
    plt.close(fig)


def popup(r):
    e = html.escape
    v = r.get('vibration')
    lines = [f"<b>{e(r['camera_id'])}</b> {e(r.get('name') or '')}, {e(str(r.get('distance_km', '')))} km"]
    if not v:
        lines.append('<i>not analyzed (no shaking detected, or no video)</i>')
        return '<br>'.join(lines)
    cam, mount = v['camera'], v.get('mount') or {}
    lines.append(f"Camera: {cam['frequency_hz']} Hz ({100 * cam['concentration']:.0f} % of the power near it), "
                 f"peak {cam['peak_px']} px, roll {cam['roll_rms_mrad']} mrad rms")
    if mount:
        lines.append(('Resonant mount' if mount.get('resonant') else 'No mount resonance')
                     + (f", rings {mount['ring_s']} s" if mount.get('ring_s') is not None else '')
                     + (f", damping {100 * mount['damping']:.1f} %" if mount.get('damping') is not None else ''))
    objs = v['objects']
    lines.append(f"{len(objs)} object{'s' if len(objs) != 1 else ''} vibrating on their own"
                 + (': ' + ', '.join(f"{o['id']}: {o['frequency_hz']} Hz" for o in objs[:6]) if objs else ''))
    plot = e(v['plot'])
    lines.append(f'<a href="{plot}" target="_blank"><img src="{plot}" width="460"></a>')
    return '<br>'.join(lines)


def map_html(rows, event, out):
    """vibration.html: the cameras colored by their vibration, the figures in the popups, the summary"""
    points = [dict(lat=r['lat'], lon=r['lon'], color=CLASS_COLORS[class_of(r)], label=r['camera_id'],
                   popup=popup(r)) for r in rows if r.get('lat') is not None]
    epi = [event['lat'], event['lon']] if event.get('lat') is not None else None
    title = html.escape(title_of(event) + ': vibration')
    legend = ''.join(f'<div><span style="background:{c}"></span>{html.escape(k)}</div>' for k, c in CLASS_COLORS.items())
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .box{{background:#fff;padding:6px 8px;font:12px sans-serif;color:{INK}}}
.box span{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px}}
.leaflet-popup-content{{font:12px/1.45 sans-serif}} .leaflet-popup-content img{{max-width:100%;height:auto}}
{PANEL_CSS}
@media (max-width:768px){{.box,.leaflet-popup-content{{font-size:14px}} .box img{{width:200px}}}}</style>
</head><body><div id="map"></div><script>
const points = {json.dumps(points)};
const epi = {json.dumps(epi)};
const map = L.map('map');
{BASEMAPS_JS}{PANEL_JS}const bounds = [];
for (const p of points) {{
  L.circleMarker([p.lat, p.lon], {{radius: 8, color: '#222', weight: 1, fillColor: p.color, fillOpacity: 0.9}})
    .bindPopup(p.popup, {{maxWidth: Math.min(480, window.innerWidth - 60)}}).bindTooltip(p.label).addTo(map);
  bounds.push([p.lat, p.lon]);
}}
if (epi) {{ L.marker(epi).bindPopup({json.dumps(title)}).addTo(map); bounds.push(epi); }}
{FIT_JS}const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{ const d = L.DomUtil.create('div', 'box');
  d.innerHTML = '<div class="ph">Legend</div><div class="pb">' + {json.dumps(legend)} + '</div>';
  return foldable(d, !small); }};
legend.addTo(map);
const info = L.control({{position: 'topleft'}});
info.onAdd = () => {{ const d = L.DomUtil.create('div', 'box');
  d.innerHTML = '<div class="ph">Vibration (for scientific use)</div><div class="pb">' +
    '<a href="vibration_summary.png" target="_blank"><img src="vibration_summary.png" width="300"></a><br>' +
    '<a href="map.html">the survey map</a></div>'; return foldable(d, !small); }};
info.addTo(map);
</script></body></html>
"""
    with open(os.path.join(out, 'vibration.html'), 'w') as f:
        f.write(page)


def vibration(out, rows, event):
    """Analyze every camera with shaking (in place: r['vibration']) and write the map and the summary"""
    for r in rows:
        try:
            r['vibration'] = analyze_camera(out, r)
        except (OSError, ValueError, np.linalg.LinAlgError) as err:
            print(f"  vibration of {r['camera_id']}: {type(err).__name__}: {err}")
            r['vibration'] = None
    done = [r for r in rows if r.get('vibration')]
    summary_png(rows, event, out)
    map_html(rows, event, out)
    print(f'vibration: {len(done)} cameras analyzed, '
          f"{sum(len(r['vibration']['objects']) for r in done)} vibrating objects -> {out}/vibration.html")
    return rows
