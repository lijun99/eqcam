"""arrivals: the P and S arrivals in the camera motion, their spectra, and the data for further analysis
(part of eqcam report --vibration)

From the motion of each camera (vibration.npz: the shaking and its ring-down at the frame rate of the video):
one-frame glitches are removed, the motion is band-passed (0.5-12 Hz) and its energy (velocity squared)
computed. The onsets are the peaks of the rise of the log energy over 0.3 s (up to the peak of the motion),
each timed where its rise starts. P and S are the pair of onsets whose interval fits the expected S-P time
(down to 25 % shorter: P starts small and is picked where it clears the noise, often late), with P clear of the
noise (SNR >= 3) and S stronger than P; without such a pair, S is the strongest onset. The stream delay is the S
pick minus its predicted arrival; P minus its own is given too, and its excess over the S delay is how late P
was picked. Each camera is labeled "P and S", "S only" or "ambiguous" (a weak onset, a view that starts during
the shaking, or a first arrival that does not fit).

Writes per camera cameras/<id>/arrivals.png, the map arrivals.html (the cameras by the quality of their picks,
the plots and the data in the popups) and arrivals_summary.png (S-P against the distance; the stream delays),
and the data in vibration_data/ (and vibration_data.zip): per camera the motion (CSV: video time, UTC, across,
down, roll, scale) and the spectra (CSV: the whole window, the noise, P and S windows), cameras.json (the
cameras, their mounts, the picks and the delays) and a README.

The camera motion is the camera's (its mount, its zoom), not the ground's: P is weak where it arrives steeply
(mostly vertical), and the spectra are the mount's response (see eqcam.frames.mount_response), not the ground
motion's: at the 2026-11-20 M3.9 event their shapes did not match those of the nearby seismic stations, even 130 m
away. And the picks are uncertain by about 1 s or more: the motion has several onsets, and which pair is P and S
depends on the expected S-P (the stream delays from the picks and from the envelopes cross-correlated with the
stations differed by 0.5-1.5 s). Use the mount frequencies, and the picks as rough times; not the spectra as
ground motion.
"""

import csv
import datetime as dt
import html
import json
import math
import os
import shutil

import matplotlib
import numpy as np
from scipy import ndimage, signal

from .frames import VP, VS, parse_time
from .report import BASEMAPS_JS, FIT_JS, PANEL_CSS, PANEL_JS, event_popup, title_of

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, MUTED = '#0b0b0b', '#52514e', '#e4e3df', '#8a8f98'
QUALITY_COLORS = {'P and S': SERIES[2], 'S only': SERIES[0], 'ambiguous': SERIES[3], 'not analyzed': MUTED}
P_COLOR, S_COLOR = SERIES[2], SERIES[7]

MIN_S_RISE = 1.0        # S: the log energy rises by this much over 0.3 s (a factor e)
P_SNR = 3.0             # P: its rms velocity over the noise's
SP_SHORT = 0.25         # P kept if S-P is at most this fraction (or 0.35 s) shorter than expected (P picked late)
SP_LONG = 0.15          # ... or at most this fraction (or 0.35 s) longer
P_LATE_NOTE_S = 0.3     # the P delay exceeds the S delay by more: noted (P picked late)


def despike(x, factor=8.0):
    """The motion (n x 2) with one-frame glitches replaced by the mean of their neighbors: a frame far from both
    neighbors while they agree (a fast real oscillation, a few frames per cycle, has neighbors that differ)"""
    y = x.copy()
    for i in range(x.shape[1]):
        a, b, c = x[:-2, i], x[1:-1, i], x[2:, i]
        d = b - (a + c) / 2
        mad = 1.4826 * np.median(np.abs(d)) + 1e-9
        bad = np.flatnonzero((np.abs(d) > factor * mad) & (np.abs(c - a) < 0.25 * np.abs(d))) + 1
        y[bad, i] = (x[bad - 1, i] + x[bad + 1, i]) / 2
    return y


def expected_sp(m, event):
    hypo = math.hypot(m.get('distance_km') or 0.0, (event or {}).get('depth_km') or 10.0)
    return hypo, hypo / VS - hypo / VP


def predicted(m, event):
    """The predicted P and S arrivals (s of the video) with the current VP and VS (a run prepared with other
    velocities is brought up to date), or those of prepare"""
    t0, origin = parse_time(m.get('start_utc')), parse_time((event or {}).get('origin_utc'))
    if t0 is None or origin is None or m.get('distance_km') is None:
        return m.get('predicted_p_s'), m.get('predicted_s_s')
    hypo, _ = expected_sp(m, event)
    o = (origin - t0).total_seconds()
    return round(o + hypo / VP, 2), round(o + hypo / VS, 2)


def pick(t, x, fps, m, event):
    """The P and S picks in the camera motion x (n x 2, px) at times t (s of the video)"""
    hypo, sp = expected_sp(m, event)
    hi = min(0.45 * fps, 12.0)
    b, a = signal.butter(2, [0.5, hi], btype='band', fs=fps)
    xf = signal.filtfilt(b, a, despike(x - x[:max(2, int(fps))].mean(0)), axis=0)
    e = np.sum(np.gradient(xf, axis=0) ** 2, axis=1) * fps ** 2        # velocity squared, (px/s)^2
    w = max(2, int(0.3 * fps))
    le = np.log(ndimage.uniform_filter1d(e, w) + 1e-12)
    rise = np.full(len(e), -np.inf)
    rise[w:] = le[w:] - le[:-w]
    env = ndimage.uniform_filter1d(e, max(2, int(fps)))
    i_max = int(np.argmax(env))
    lo = int(fps)
    out = dict(hypo_km=round(hypo, 1), sp_expected_s=round(sp, 2), quality='ambiguous', s_s=None, p_s=None,
               p_snr=None, s_rise=None, delay_s_s=None, delay_p_s=None, p_late_s=None, notes=[])
    if i_max <= lo:
        out['notes'].append('the motion peaks at the start of the window')
        return out, xf, e
    noise = float(e[:int(fps)].mean()) + 1e-12
    fits = lambda d: sp - max(0.35, SP_SHORT * sp) <= d <= sp + max(0.35, SP_LONG * sp)     # noqa: E731
    level = lambda i, j: float(np.sqrt(e[i:j].mean() / noise)) if j > i else 0.0   # noqa: E731
    def onset(k):
        """The onset of the rise peaking at k: where the rise reaches half its peak (the energy is smoothed over
        a centered window, so this is close to where the motion starts)"""
        j = k
        while j > lo and rise[j - 1] > 0.5 * rise[k]:
            j -= 1
        return j

    # the onsets: the peaks of the rise of the log energy, up to the peak of the motion
    seg = np.where(np.isfinite(rise[lo:i_max + int(0.5 * fps) + 1]), rise[lo:i_max + int(0.5 * fps) + 1], -1)
    peaks = lo + signal.find_peaks(seg, height=0.5, distance=max(1, int(0.5 * fps)))[0]
    i_c = int(peaks[np.argmax(rise[peaks])]) if len(peaks) else lo + int(np.argmax(seg))   # the strongest
    # a pair of onsets at the expected S-P time: P clear of the noise, S stronger than P
    best, i_p, i_s = None, None, i_c
    for i in peaks:
        for j in peaks[peaks > i]:
            js = onset(j)                                   # P's level up to where S starts, not into S
            if fits(t[j] - t[i]) and rise[j] >= MIN_S_RISE and level(i, js) >= P_SNR \
                    and level(j, j + int(2 * fps)) > 1.5 * level(i, js):
                score = rise[i] + rise[j]
                if best is None or score > best:
                    best, i_p, i_s = score, int(i), int(j)
    if i_p is None:
        strong = [int(k) for k in peaks if k != i_c and rise[k] >= 0.7 * rise[i_c] and abs(t[k] - t[i_c]) <= 1.5 * sp]
        if strong:
            out['quality'] = 'ambiguous'
            out['notes'].append('another strong onset at ' + ', '.join(f'{t[k]:.2f}' for k in strong)
                                + ' s, not at the S-P time')
        else:
            out['quality'] = 'S only'
    out.update(s_s=round(float(t[onset(i_s)]), 2), s_rise=round(float(rise[i_s]), 2))
    if i_p is not None:
        out.update(p_s=round(float(t[onset(i_p)]), 2), p_snr=round(level(i_p, onset(i_s)), 1), quality='P and S')
    if rise[i_s] < MIN_S_RISE:
        out['quality'] = 'ambiguous'
        out['notes'].append('the S onset is not sharp')
    if 'no_calm_frames_before_the_shaking' in (m.get('flags') or []) or i_s < 1.5 * fps:
        out['quality'] = 'ambiguous'
        out['notes'].append('the view starts during the shaking')
    if out['quality'] != 'ambiguous':
        out['delay_s_s'] = round(out['s_s'] - m['predicted_s_s'], 2) if m.get('predicted_s_s') is not None else None
        if out['p_s'] is not None and m.get('predicted_p_s') is not None:
            out['delay_p_s'] = round(out['p_s'] - m['predicted_p_s'], 2)
            if out['delay_s_s'] is not None:
                out['p_late_s'] = round(out['delay_p_s'] - out['delay_s_s'], 2)
                if out['p_late_s'] > P_LATE_NOTE_S:
                    out['notes'].append(f"P picked {out['p_late_s']:.1f} s later than the S delay predicts (P is "
                                        'small at its onset, or the pair is not P and S); the delay is from S')
    return out, xf, e


NFFT_S = 8      # s: the frequency grid of the spectra (0.125 Hz)


def spectra(xf, fps, picks, t):
    """The spectra (px^2/Hz, both axes summed) of the whole window and of the noise, P and S windows, on one
    frequency grid"""
    nfft = int(2 ** math.ceil(math.log2(NFFT_S * fps)))
    f, full = signal.welch(xf, fps, nperseg=min(len(xf), nfft), nfft=nfft, axis=0)
    out = dict(freq=f, full=full.sum(1))
    idx = lambda s: None if s is None else int(np.searchsorted(t, s))  # noqa: E731
    i_s, i_p = idx(picks['s_s']), idx(picks['p_s'])
    first = i_p if i_p is not None else i_s
    windows = dict(noise=(0, first), p=(i_p, i_s), s=(i_s, None if i_s is None else i_s + int(3 * fps)))
    for name, (a, b) in windows.items():
        if a is None or b is None or b - a < 8:
            out[name] = np.full(len(f), np.nan)
            continue
        seg = xf[a:b]
        _, p = signal.periodogram(seg - seg.mean(0), fps, window='hann', nfft=nfft, axis=0)
        out[name] = p.sum(1)
    return out


def plot(m, t, xf, picks, spec, path, fps):
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2.2, 1]},
                                  constrained_layout=True)
    for i, (lab, col) in enumerate((('across', SERIES[0]), ('down', SERIES[1]))):
        ax.plot(t, xf[:, i], color=col, lw=0.9, label=lab)
    for key, lab, col, y in (('p_s', 'P', P_COLOR, 0.86), ('s_s', 'S', S_COLOR, 0.97)):   # labels staggered
        if picks[key] is not None:
            ax.axvline(picks[key], color=col, lw=2)
            ax.text(picks[key], y, f' {lab} {picks[key]:.2f} s', transform=ax.get_xaxis_transform(), color=INK,
                    fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, lw=1))
    if picks['delay_s_s'] is not None:                       # the predicted arrivals, with the delay from S
        for key, col in (('predicted_p_s', P_COLOR), ('predicted_s_s', S_COLOR)):
            ax.axvline(m[key] + picks['delay_s_s'], color=col, lw=1, ls=':')
    ax.legend(frameon=False, fontsize=11, loc='lower right')
    sp_obs = (f"{picks['s_s'] - picks['p_s']:.2f} s" if picks['p_s'] is not None and picks['s_s'] is not None else '-')
    ax.set_title(f"{m['camera_id']} {m.get('name', '')}: {picks['quality']}; S-P {sp_obs} "
                 f"(expected {picks['sp_expected_s']} s)\ndotted: the predicted P and S with the stream delay from S",
                 fontsize=12, color=INK, loc='left')
    ax.set_xlabel('time in the video, s', fontsize=12, color=INK2)
    ax.set_ylabel('camera motion, px (0.5-12 Hz)', fontsize=12, color=INK2)
    for name, lab, col in (('noise', 'before the arrivals', MUTED), ('p', 'P window', P_COLOR), ('s', 'S window', S_COLOR)):
        p = spec[name]
        if np.isfinite(p).any():
            axf.semilogy(spec['freq'][1:], p[1:] + 1e-12, color=col, lw=1.4, label=lab)
    axf.set_xlim(0, min(0.45 * fps, 12))
    axf.legend(frameon=False, fontsize=11)
    axf.set_title("camera motion spectra (the mount's response,\nnot the ground motion's)", fontsize=12, color=INK,
                  loc='left')
    axf.set_xlabel('frequency, Hz', fontsize=12, color=INK2)
    axf.set_ylabel('px² / Hz', fontsize=12, color=INK2)
    for a in (ax, axf):
        a.grid(color=GRID, lw=0.6)
        a.tick_params(labelsize=11, colors=INK2)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    fig.savefig(path, dpi=90, facecolor='white')
    plt.close(fig)


def write_data(out, m, event, z, picks, spec):
    """The motion and the spectra of one camera as CSV in vibration_data/"""
    ddir = os.path.join(out, 'vibration_data')
    os.makedirs(ddir, exist_ok=True)
    cid = m['camera_id']
    t0 = parse_time(m.get('start_utc'))
    roll = z['roll'] if 'roll' in z else np.full(len(z['t']), np.nan)
    zoom = z['scale'] if 'scale' in z else np.full(len(z['t']), np.nan)
    with open(os.path.join(ddir, f'{cid}_motion.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video_time_s', 'utc_received', 'camera_across_px', 'camera_down_px', 'roll_mrad', 'scale_ppm'])
        for k, tk in enumerate(z['t']):
            utc = (t0 + dt.timedelta(seconds=float(tk))).isoformat(timespec='milliseconds').replace('+00:00', 'Z') \
                if t0 else ''
            w.writerow([f'{tk:.4f}', utc, f"{z['camera_px'][k, 0]:.5f}", f"{z['camera_px'][k, 1]:.5f}",
                        f'{roll[k] * 1000:.5f}', f'{zoom[k] * 1e6:.3f}'])
    if len(z['object_ids']):
        with open(os.path.join(ddir, f'{cid}_objects.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video_time_s'] + [f'object_{o}_{c}_px' for o in z['object_ids'] for c in ('across', 'down')])
            for k, tk in enumerate(z['t']):
                w.writerow([f'{tk:.4f}'] + [f"{z['object_px'][k, i, c]:.5f}" for i in range(len(z['object_ids']))
                                            for c in (1, 0)])
    with open(os.path.join(ddir, f'{cid}_spectra.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frequency_hz', 'psd_whole_px2_per_hz', 'psd_noise', 'psd_p_window', 'psd_s_window'])
        for k, fk in enumerate(spec['freq']):
            w.writerow([f'{fk:.4f}'] + ['' if not np.isfinite(spec[n][k]) else f'{spec[n][k]:.6e}'
                                        for n in ('full', 'noise', 'p', 's')])


README = """Camera vibration data of an earthquake, from eqcam (https://github.com/lijun99/eqcam)

event.json        the event (origin time, epicenter, depth, magnitude)
cameras.json      per camera: location, distances, video frame rate, pixel scale, view, the mount response, the
                  P and S picks, their quality, the stream delay (from S; p_late_s: how late P was picked by
                  the same measure), the predicted arrivals (Vp 5.7, Vs 3.1 km/s), the files below
<id>_motion.csv   the motion of the camera, from 5 s before the shaking to 15 s after it, in the view of the
                  shaking: video_time_s (s from the first frame of the video), utc_received (UTC of the frame as
                  received by the recorder: later than the ground motion by the stream delay), camera_across_px,
                  camera_down_px (the shift of the image, pixels of the full frame, relative to the first frame
                  of this series), roll_mrad (rotation about the image center), scale_ppm (zoom-like change)
<id>_objects.csv  if any: the motion of objects vibrating on their own, relative to the camera (px)
<id>_spectra.csv  power spectra of the camera motion (px^2/Hz, both axes summed, band-passed 0.5-12 Hz): the
                  whole series, the noise before the arrivals, the P window (P to S), the S window (3 s from S)

Caveats: the motion is the camera's, not the ground's: it is the ground motion filtered by the camera's mount
(a pole, a mast, a building: see "mount" in cameras.json, a resonance frequency, ring-down and damping) and
scaled by the zoom (pixels, not ground units). The spectra are therefore the mount's response, not ground-motion
spectra: compared with nearby seismic stations at one event (2026-11-20, M3.9, Los Angeles), their shapes and
their S/P ratios did not match, even at a station 130 m from the camera. The frame rate limits the frequencies
(up to half of it). The picks are uncertain by about 1 s or more: the camera motion has several onsets, P starts
small, and at that event the camera S-P times were 0.3-1 s shorter than the stations', and the stream delays from
the picks and from the envelopes cross-correlated with the stations differed by 0.5-1.5 s. The times are those of
the frames as received: subtract the stream delay (cameras.json, from S, when available) to compare with seismic
data; an independent clock (e.g., one burned into the video) is better. Automated and unverified.
"""


def popup(r):
    e = html.escape
    a = r.get('arrivals')
    lines = [f"<b>{e(r['camera_id'])}</b> {e(r.get('name') or '')}, {e(str(r.get('distance_km', '')))} km"]
    if not a:
        lines.append('<i>not analyzed (no camera vibration detected, or no video)</i>')
        return '<br>'.join(lines)
    sp = f"{a['s_s'] - a['p_s']:.2f} s" if a['p_s'] is not None and a['s_s'] is not None else '-'
    lines.append(f"<b>{e(a['quality'])}</b>: P {a['p_s'] if a['p_s'] is not None else '-'} s, "
                 f"S {a['s_s'] if a['s_s'] is not None else '-'} s (video time); S-P {sp}, expected "
                 f"{a['sp_expected_s']} s")
    if a['delay_s_s'] is not None:
        lines.append(f"stream delay: {a['delay_s_s']} s (from S)"
                     + (f"; P picked {a['p_late_s']:+.2f} s from its prediction with that delay"
                        if a.get('p_late_s') is not None else ''))
    lines.append("<small>The spectra are the camera mount's response, not the ground motion's; the picks are "
                 'uncertain by about 1 s or more.</small>')
    if a['notes']:
        lines.append(f"<small>{e('; '.join(a['notes']))}</small>")
    plot_path = e(a['plot'])
    lines.append(f'<a href="{plot_path}" target="_blank"><img src="{plot_path}" width="460"></a>')
    files = [f"{r['camera_id']}_motion.csv", f"{r['camera_id']}_spectra.csv"]
    lines.append('Data: ' + ' &middot; '.join(f'<a href="vibration_data/{e(f)}">{e(f)}</a>' for f in files))
    return '<br>'.join(lines)


def summary_png(rows, event, out):
    done = [r for r in rows if r.get('arrivals')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    if done:
        hypos = [r['arrivals']['hypo_km'] for r in done]
        h = np.linspace(0, max(hypos) * 1.1, 50)
        a1.plot(h, h / VS - h / VP, color=MUTED, lw=1.2, label=f'expected (Vp {VP:g}, Vs {VS:g} km/s)')
    for r in done:
        a = r['arrivals']
        if a['p_s'] is not None and a['s_s'] is not None:
            a1.scatter(a['hypo_km'], a['s_s'] - a['p_s'], s=70, color=QUALITY_COLORS[a['quality']], ec='white',
                       lw=1.5, zorder=3)
            a1.annotate(r['camera_id'], (a['hypo_km'], a['s_s'] - a['p_s']), xytext=(6, 4),
                        textcoords='offset points', fontsize=11, color=INK)
    a1.legend(frameon=False, fontsize=11, loc='upper left')
    a1.set_xlabel('hypocentral distance, km', fontsize=12, color=INK2)
    a1.set_ylabel('S-P time, s', fontsize=12, color=INK2)
    a1.set_title('S-P time (P and S picked)', fontsize=12, color=INK2, loc='left')
    with_delay = sorted((r for r in done if r['arrivals']['delay_s_s'] is not None),
                        key=lambda r: r['arrivals']['delay_s_s'])
    labeled = set()
    for i, r in enumerate(with_delay):
        a = r['arrivals']
        a2.scatter(a['delay_s_s'], i, s=70, color=S_COLOR, ec='white', lw=1.5, zorder=3,
                   label=None if 'S' in labeled else 'from S')
        labeled.add('S')
        if a['delay_p_s'] is not None:
            a2.scatter(a['delay_p_s'], i, s=50, marker='D', color=P_COLOR, ec='white', lw=1, zorder=3,
                       label=None if 'P' in labeled else 'from P (P often picked late)')
            labeled.add('P')
    a2.set_yticks(range(len(with_delay)))
    a2.set_yticklabels([f"{r['camera_id']} {r.get('name', '')[:18]}" for r in with_delay], fontsize=11)
    if with_delay:
        a2.legend(frameon=False, fontsize=11, loc='lower right')
    a2.set_xlabel('stream delay implied, s (pick - predicted arrival)', fontsize=12, color=INK2)
    a2.set_title('stream delays', fontsize=12, color=INK2, loc='left')
    for a in (a1, a2):
        a.grid(color=GRID, lw=0.6)
        a.tick_params(labelsize=11, colors=INK2)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    fig.suptitle(f'{title_of(event)}: P and S arrivals in the camera motion', fontsize=13, color=INK, x=0.01, ha='left')
    fig.savefig(os.path.join(out, 'arrivals_summary.png'), dpi=90, facecolor='white')
    plt.close(fig)


def map_html(rows, event, out):
    points = []
    for r in rows:
        if r.get('lat') is None:
            continue
        q = (r.get('arrivals') or {}).get('quality', 'not analyzed')
        points.append(dict(lat=r['lat'], lon=r['lon'], color=QUALITY_COLORS[q], label=r['camera_id'], popup=popup(r)))
    epi = [event['lat'], event['lon']] if event.get('lat') is not None else None
    title = html.escape(title_of(event) + ': arrivals')
    legend = ''.join(f'<div><span style="background:{c}"></span>{html.escape(k)}</div>' for k, c in QUALITY_COLORS.items())
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
    .bindPopup(p.popup, popupOptions(480)).bindTooltip(p.label).addTo(map);
  bounds.push([p.lat, p.lon]);
}}
if (epi) {{ L.marker(epi).bindPopup({json.dumps(event_popup(event))}).addTo(map); bounds.push(epi); }}
{FIT_JS}const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{ const d = L.DomUtil.create('div', 'box');
  d.innerHTML = '<div class="ph">P and S picks</div><div class="pb">' + {json.dumps(legend)} + '</div>';
  return foldable(d, !small); }};
legend.addTo(map);
const info = L.control({{position: 'topleft'}});
info.onAdd = () => {{ const d = L.DomUtil.create('div', 'box');
  d.innerHTML = '<div class="ph">Arrivals (for scientific use)</div><div class="pb">' +
    '<a href="arrivals_summary.png" target="_blank"><img src="arrivals_summary.png" width="320"></a><br>' +
    '<a href="vibration_data.zip">all the data (zip)</a> &middot; <a href="vibration_data/README.txt">README</a>' +
    '<br><a href="vibration.html">the vibration map</a> &middot; <a href="map.html">the survey map</a></div>';
  return foldable(d, !small); }};
info.addTo(map);
</script></body></html>
"""
    with open(os.path.join(out, 'arrivals.html'), 'w') as f:
        f.write(page)


def arrivals(out, rows, event):
    """Pick every camera analyzed by eqcam.vibration (in place: r['arrivals']), write the plots, the data, the
    map and the summary"""
    ddir = os.path.join(out, 'vibration_data')
    shutil.rmtree(ddir, ignore_errors=True)
    os.makedirs(ddir)
    cameras = []
    for r in rows:
        r['arrivals'] = None
        path = os.path.join(out, 'cameras', r['camera_id'], 'vibration.npz')
        if not r.get('vibration') or not os.path.exists(path):
            continue
        z = np.load(path)
        r['predicted_p_s'], r['predicted_s_s'] = predicted(r, event)
        t, x = z['t'], z['camera_px'].astype(float)
        fps = float(r['vibration'].get('fps') or 1 / np.median(np.diff(t)))
        picks, xf, _ = pick(t, x, fps, r, event)
        spec = spectra(xf, fps, picks, t)
        cpath = os.path.join(out, 'cameras', r['camera_id'], 'arrivals.png')
        plot(r, t, xf, picks, spec, cpath, fps)
        picks['plot'] = os.path.relpath(cpath, out)
        r['arrivals'] = picks
        write_data(out, r, event, z, picks, spec)
        cameras.append(dict(camera_id=r['camera_id'], name=r.get('name'), lat=r.get('lat'), lon=r.get('lon'),
                            view=r.get('view'), epicentral_km=r.get('distance_km'), hypocentral_km=picks['hypo_km'],
                            start_utc=r.get('start_utc'), video_fps=r['vibration'].get('fps'),
                            px_scale=r['vibration'].get('px_scale'), width=r.get('width'), height=r.get('height'),
                            mount=r.get('mount'), predicted_p_s=r.get('predicted_p_s'),
                            predicted_s_s=r.get('predicted_s_s'), picks={k: v for k, v in picks.items() if k != 'plot'},
                            files=sorted(f for f in os.listdir(ddir) if f.startswith(r['camera_id'] + '_'))))
    with open(os.path.join(ddir, 'cameras.json'), 'w') as f:
        json.dump(cameras, f, indent=1)
    with open(os.path.join(ddir, 'event.json'), 'w') as f:
        json.dump(event, f, indent=1)
    with open(os.path.join(ddir, 'README.txt'), 'w') as f:
        f.write(README)
    shutil.make_archive(os.path.join(out, 'vibration_data'), 'zip', out, 'vibration_data')
    summary_png(rows, event, out)
    map_html(rows, event, out)
    counts = {q: sum(1 for r in rows if (r.get('arrivals') or {}).get('quality') == q) for q in QUALITY_COLORS}
    print('arrivals: ' + ', '.join(f'{v} {k}' for k, v in counts.items() if v and k != 'not analyzed')
          + f' -> {out}/arrivals.html, vibration_data.zip')
    return rows
