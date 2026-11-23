"""calibrate: a catalog of the cameras, from clips recorded outside events

Each clip (a few minutes of a camera's stream, recorded routinely: by day and by night, on calm and windy days)
is checked and measured:

- the stream: resolution, frame rate, bitrate;
- the picture: brightness, contrast, sharpness, frozen or dark; day, twilight or night (the elevation of the sun
  at the camera at the time of the clip, else the brightness);
- the views: fixed, or touring presets (the view changes, as in prepare), and the calm jitter of the image;
- the ambient vibration of the mount (wind, traffic): the spectrum of the image motion in the stable views, at
  the native frame rate; its peaks, their prominence over the background and the half-power width of the
  strongest (its damping).

Earlier events add what they measured: the mount's frequency and damping in the shaking, and the stream delay
from the P and S arrivals (with report --vibration). Each clip's results are kept in clips/<camera_id>/
<clip_id>.json and each event's in events/<event_id>.json, so calibrating more clips or events adds to the
catalog, which is rebuilt each time: catalog.json, catalog.csv and per camera calibration/<camera_id>.png (the
ambient spectra of its clips, with the mount frequencies measured at events), and changes.json (what changed
since the catalog before: cameras new, offline or back, no longer usable, a new ambient peak).

For periodic runs, calibrate(streams=...) first records a clip of every live stream (eqcam.record) into
recordings/<UTC time>/, and deletes the videos older than keep_days; a stream that cannot be recorded counts
as offline (the uptime of the camera).

The quality bars are provisional (QUALITY below): to be agreed with the users of the catalog.
"""

import csv
import datetime as dt
import glob
import json
import math
import os
import subprocess

import numpy as np
from scipy import ndimage, signal

from .frames import (analyze_motion, decode_small, distance_km, frame_differences, image_shifts, parse_time,
                     read_cameras)
from .vibration import INK, MUTED, SERIES, highpass, style

QUALITY = dict(
    min_height_px=480,        # survey: the picture at least this tall
    min_brightness=30,        # mean gray level (0-255) below which the picture is dark
    max_frozen=0.5,           # fraction of frames identical to the one before, above which the stream froze
    min_sharpness=20.0,       # variance of the Laplacian of the 640-wide gray frame, below which it is blurred
    min_fps=15.0,             # vibration: native frame rate at least this
    min_stable_s=20.0,        # and a stable view of at least this long
    max_calm_jitter_px=0.5,   # and the calm image steadier than this (px of the native frame, frames 1/15 s apart)
)
MOTION_FPS, MOTION_WIDTH = 15.0, 320     # for the views, as in prepare
AMBIENT_WIDTH = 480                      # px, the frames tracked for the ambient vibration
AMBIENT_FPS_MAX = 30.0
NFFT_S = 8.0                             # s, the segments of the ambient spectrum (0.125 Hz)
CHUNK_S = 32.0                           # s, decoded at a time
MIN_PROMINENCE = 4.0                     # a peak: its power at least this many times the background around it
BACKGROUND_HZ = 2.0                      # the width of the running median for the background
MAX_PEAKS = 4
SUN_TWILIGHT_DEG = -6.0


# ------------------------------------------------------------------------------------------- the clip

def sun_elevation(t, lat, lon):
    """The elevation of the sun (degrees) at time t (UTC) at lat, lon (to about 0.1 degree)"""
    d = (t - dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    ecl = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    eps = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(eps) * math.sin(ecl), math.cos(ecl))
    dec = math.asin(math.sin(eps) * math.sin(ecl))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))


def stream_info(video):
    """Width, height, frame rate, bitrate (kbit/s), codec and duration of a video"""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,avg_frame_rate,codec_name:format=duration,bit_rate', '-of', 'json',
                          video], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    s, f = info['streams'][0], info.get('format', {})
    try:
        num, den = s.get('avg_frame_rate', '0/0').split('/')
        fps = float(num) / float(den) if float(den) else None
    except ValueError:
        fps = None
    rate = f.get('bit_rate')
    return dict(width=int(s['width']), height=int(s['height']), fps=None if fps is None else round(fps, 2),
                bitrate_kbps=round(int(rate) / 1000) if rate and rate.isdigit() else None,
                codec=s.get('codec_name'), duration_s=round(float(f.get('duration', 'nan')), 2))


def picture(video, height, width, start, length, n=8):
    """Brightness, RMS contrast and sharpness (variance of the Laplacian) of n frames across the clip, decoded
    640 wide in gray; the medians"""
    w = 640
    h = max(2, int(round(height * w / width / 2)) * 2)
    frames = decode_small(video, w, h, min(1.0, n / max(length, 1.0)), start, length).astype(np.float32)
    if not len(frames):
        return {}
    bright = frames.mean(axis=(1, 2))
    contrast = frames.std(axis=(1, 2)) / np.maximum(bright, 1.0)
    sharp = np.array([ndimage.laplace(f).var() for f in frames])
    return dict(brightness=round(float(np.median(bright)), 1), contrast=round(float(np.median(contrast)), 3),
                sharpness=round(float(np.median(sharp)), 1))


def peak_width(fr, p, k):
    """The half-power width (Hz) of the peak at index k, interpolated on both sides; None if it does not fall
    to half within the spectrum"""
    half = p[k] / 2
    sides = []
    for step in (-1, 1):
        j = k
        while 0 <= j + step < len(p) and p[j + step] > half:
            j += step
        if not 0 <= j + step < len(p):
            return None
        a, b = p[j], p[j + step]                    # a > half >= b
        sides.append(fr[j] + (fr[j + step] - fr[j]) * (a - half) / (a - b))
    return float(sides[1] - sides[0])


def ambient(video, info, segments, start, max_s=300.0):
    """The ambient vibration of the camera: the image shift (relative to the middle frame of each chunk of a
    stable view) at the native frame rate, high-passed at 0.5 Hz; its spectrum averaged over the chunks; the
    peaks, the strongest one's prominence, half-power width and damping"""
    fps = min(info['fps'] or 30.0, AMBIENT_FPS_MAX)
    nfft = int(round(NFFT_S * fps))
    h = max(2, int(round(info['height'] * AMBIENT_WIDTH / info['width'] / 2)) * 2)
    scale = info['width'] / AMBIENT_WIDTH
    psds, used, rms = [], 0.0, []
    for a_s, b_s in segments:
        t = a_s
        while t + NFFT_S <= b_s and used < max_s:
            length = min(CHUNK_S, b_s - t)
            frames = decode_small(video, AMBIENT_WIDTH, h, fps, start + t, length)
            t += length
            if len(frames) < nfft:
                continue
            shifts, q = image_shifts(frames, len(frames) // 2, heights=True)
            if np.median(q) <= 0 or (q < 0.3 * np.median(q)).mean() > 0.05 or \
                    np.abs(np.diff(shifts, axis=0)).max() > 8:          # not one steady view
                continue
            x = highpass(shifts.astype(np.float64) * scale, fps)
            fr, p = signal.welch(x, fps, nperseg=nfft, axis=0)
            psds.append((p.sum(axis=1), len(frames)))
            band = (fr >= 0.5) & (fr <= 0.4 * fps)
            rms.append(float(np.sqrt((p.sum(axis=1)[band]).sum() * (fr[1] - fr[0]))))
            used += len(frames) / fps
    if not psds:
        return None
    p = sum(w * s for s, w in psds) / sum(w for _, w in psds)
    band = (fr >= 0.5) & (fr <= 0.4 * fps)
    pb, fb = p[band], fr[band]
    df = float(fr[1] - fr[0])
    # the background under the peaks: a running median of the log spectrum over 2 Hz (the spectrum falls
    # steeply with frequency: against its overall median, any low frequency would stand out)
    size = max(3, int(round(BACKGROUND_HZ / df)) | 1)
    base = np.exp(ndimage.median_filter(np.log(pb + 1e-30), size, mode='nearest'))
    ks, _ = signal.find_peaks(pb / base, height=MIN_PROMINENCE)
    ks = sorted(ks, key=lambda k: -pb[k] / base[k])[:MAX_PEAKS]
    out = dict(fps=round(fps, 2), seconds=round(used, 1), rms_px=round(float(np.median(rms)), 4),
               peaks=[dict(frequency_hz=round(float(fb[k]), 2), prominence=round(float(pb[k] / base[k]), 1))
                      for k in ks],
               spectrum=dict(frequency_hz=[round(float(f), 3) for f in fr],
                             psd_px2_per_hz=[float('%.4g' % v) for v in p]))
    if ks:
        k = ks[0]
        f0 = float(fb[k])
        width = peak_width(fb, pb, k)
        near = np.abs(fb - f0) <= max(0.25, 0.1 * f0)
        out.update(frequency_hz=round(f0, 2), prominence=round(float(pb[k] / base[k]), 1),
                   concentration=round(float(pb[near].sum() / pb.sum()), 2),
                   width_hz=None if width is None else round(width, 3),
                   damping=None if width is None else round(width / (2 * f0), 3),
                   width_resolved=bool(width is not None and width > 2 * df))
    return out


def calibrate_clip(row, videos):
    """The quality, views and ambient vibration of one clip"""
    video = os.path.join(videos, row['video']) if row.get('video') else ''
    r = dict(camera_id=row['camera_id'], clip_id=row['clip_id'], video=os.path.basename(video),
             start_utc=row.get('start_utc') or None, wind_mps=float(row['wind_mps']) if row.get('wind_mps') else None,
             wind_station=row.get('wind_station') or None,
             problems=[], vibration_problems=[])
    if row.get('error'):                                # the stream could not be recorded
        r.update(problems=['offline'], error=row['error'])
        return r
    if not video or not os.path.exists(video):
        r['problems'].append('no video')
        return r
    info = stream_info(video)
    r['stream'] = info
    start = float(row['start_s']) if row.get('start_s') else 0.0
    end = float(row['end_s']) if row.get('end_s') else info['duration_s']
    length = max(0.0, end - start)
    r.update(start_s=round(start, 2), end_s=round(end, 2))

    # the views and the calm jitter
    h = max(2, int(round(info['height'] * MOTION_WIDTH / info['width'] / 2)) * 2)
    frames = decode_small(video, MOTION_WIDTH, h, MOTION_FPS, start, length)
    if len(frames) < 3 * MOTION_FPS:
        r['problems'].append('too short')
        return r
    d = frame_differences(frames)
    r['frozen_fraction'] = round(float((d < 0.02).mean()), 3)
    sh = analyze_motion(frames, MOTION_FPS)
    scale = info['width'] / MOTION_WIDTH
    stable = np.zeros(len(frames), bool)
    for a, b in sh['segments']:
        stable[a:b] = True
    segments = [(a / MOTION_FPS, b / MOTION_FPS) for a, b in sh['segments']]
    r.update(view_changes=len(sh['changes']), stable_fraction=round(float(stable.mean()), 3),
             longest_stable_s=round(max((b - a for a, b in segments), default=0.0), 1),
             calm_jitter_px=round(float(np.median(sh['jitter'][stable])) * scale, 4) if stable.any() else None)
    r['picture'] = picture(video, info['height'], info['width'], start, length)

    # day or night
    t0 = parse_time(row.get('start_utc'))
    if t0 and row.get('lat') is not None and row.get('lon') is not None:
        el = sun_elevation(t0 + dt.timedelta(seconds=start + length / 2), row['lat'], row['lon'])
        r['sun_elevation_deg'] = round(el, 1)
        r['light'] = 'day' if el >= 0 else 'twilight' if el >= SUN_TWILIGHT_DEG else 'night'
    else:
        r['light'] = 'day' if r['picture'].get('brightness', 0) >= 80 else 'night'
        r['light_from'] = 'brightness'

    # the bars
    pic = r['picture']
    if info['height'] < QUALITY['min_height_px']:
        r['problems'].append(f"low resolution ({info['width']} x {info['height']})")
    if pic.get('brightness', 0) < QUALITY['min_brightness']:
        r['problems'].append('dark')
    if r['frozen_fraction'] > QUALITY['max_frozen']:
        r['problems'].append('frozen')
    if pic.get('sharpness', 0) < QUALITY['min_sharpness']:
        r['problems'].append('blurred or hazy')
    if (info['fps'] or 0) < QUALITY['min_fps']:
        r['vibration_problems'].append(f"low frame rate ({info['fps']} fps)")
    if r['longest_stable_s'] < QUALITY['min_stable_s']:
        r['vibration_problems'].append('no long stable view')
    if r['calm_jitter_px'] is not None and r['calm_jitter_px'] > QUALITY['max_calm_jitter_px']:
        r['vibration_problems'].append('unsteady')

    r['ambient'] = ambient(video, info, segments, start) if info['fps'] else None
    return r


# ----------------------------------------------------------------------------------------- the events

def event_record(run):
    """What an event's run measured per camera: the mount response in the shaking, and the stream delays from
    the P and S arrivals (report --vibration)"""
    event = json.load(open(os.path.join(run, 'event.json'))) if os.path.exists(os.path.join(run, 'event.json')) else {}
    metrics = json.load(open(os.path.join(run, 'metrics.json')))
    picks = {}
    path = os.path.join(run, 'vibration_data', 'cameras.json')
    if os.path.exists(path):
        picks = {c['camera_id']: c.get('picks') or {} for c in json.load(open(path))}
    cams = {}
    for m in metrics:
        c = dict(name=m.get('name', ''), lat=m.get('lat'), lon=m.get('lon'), view=m.get('view', ''),
                 shaking_detected=m.get('shaking_detected'), mount=m.get('mount'))
        if event.get('lat') is not None and m.get('lat') is not None:
            c['distance_km'] = round(distance_km(event['lat'], event['lon'], m['lat'], m['lon']), 1)
        p = picks.get(m['camera_id'])
        if p:
            c.update(arrival_quality=p.get('quality'), stream_delay_s=p.get('delay_s_s'),    # from S: P is often late
                     p_late_s=p.get('p_late_s'))
        cams[m['camera_id']] = c
    eid = event.get('event_id') or os.path.basename(os.path.dirname(os.path.abspath(run)))
    return dict(event_id=eid, magnitude=event.get('magnitude'), origin_utc=event.get('origin_utc'), cameras=cams)


# ---------------------------------------------------------------------------------------- the catalog

def _median(values, nd=2):
    v = [x for x in values if x is not None]
    return round(float(np.median(v)), nd) if v else None


def near(f, g):
    return abs(f - g) <= max(0.25, 0.1 * min(f, g))


def group_peaks(ambients):
    """The ambient peaks of all the clips, grouped by frequency: the mean frequency, the number of clips with
    the peak and the median prominence; most frequent first"""
    groups = []
    for a in ambients:
        for p in a.get('peaks') or []:
            g = next((g for g in groups if near(np.mean(g['f']), p['frequency_hz'])), None)
            if g is None:
                groups.append(g := dict(f=[], prom=[]))
            g['f'].append(p['frequency_hz'])
            g['prom'].append(p['prominence'])
    out = [dict(frequency_hz=round(float(np.mean(g['f'])), 2), clips=len(g['f']),
                prominence=round(float(np.median(g['prom'])), 1)) for g in groups]
    return sorted(out, key=lambda p: (-p['clips'], -p['prominence']))


def summarize(cam, clips, events):
    """One camera's entry of the catalog"""
    s = dict(camera_id=cam['camera_id'], name=cam.get('name', ''), lat=cam.get('lat'), lon=cam.get('lon'),
             view=cam.get('view', ''), clips=len(clips))
    with_stream = [c for c in clips if c.get('stream')]
    if clips:
        latest = max(clips, key=lambda c: c.get('start_utc') or '')
        s.update(offline_clips=sum('offline' in c['problems'] for c in clips),
                 uptime=round(len(with_stream) / len(clips), 2), latest_clip_utc=latest.get('start_utc'),
                 latest_problems=latest['problems'])
    if with_stream:
        last = max(with_stream, key=lambda c: c.get('start_utc') or '')['stream']
        s['stream'] = {k: last[k] for k in ('width', 'height', 'fps', 'bitrate_kbps', 'codec')}
    for light in ('day', 'twilight', 'night'):
        cs = [c for c in with_stream if c.get('light') == light]
        if not cs:
            s[light] = None
            continue
        good = [c for c in cs if not c['problems']]
        problems = sorted({p for c in cs for p in c['problems']})
        s[light] = dict(clips=len(cs), usable=len(good) / len(cs) >= 0.5, problems=problems,
                        brightness=_median([c['picture'].get('brightness') for c in cs], 1),
                        sharpness=_median([c['picture'].get('sharpness') for c in cs], 1))
    s['touring'] = any((c.get('view_changes') or 0) > 0 for c in with_stream) if with_stream else None
    s['calm_jitter_px'] = _median([c.get('calm_jitter_px') for c in with_stream], 4)
    vib = [c for c in with_stream if not c['vibration_problems']]
    s['vibration_usable'] = bool(vib) if with_stream else None
    s['vibration_problems'] = sorted({p for c in with_stream for p in c['vibration_problems']})
    amb = [c['ambient'] for c in with_stream if c.get('ambient')]
    clear = [a for a in amb if a.get('frequency_hz') is not None]
    if amb:
        freqs = [a['frequency_hz'] for a in clear]
        s['ambient'] = dict(clips=len(amb), with_peak=len(clear), frequency_hz=_median(freqs),
                            frequency_spread_hz=round(float(np.ptp(freqs)), 2) if freqs else None,
                            damping=_median([a.get('damping') for a in clear if a.get('width_resolved')], 3),
                            rms_px=_median([a.get('rms_px') for a in amb], 4), peaks=group_peaks(amb))
    else:
        s['ambient'] = None
    ev = []
    for e in events:
        c = e['cameras'].get(cam['camera_id'])
        if c:
            mount = c.get('mount') or {}
            ev.append(dict(event_id=e['event_id'], magnitude=e.get('magnitude'), distance_km=c.get('distance_km'),
                           shaking_detected=c.get('shaking_detected'), mount_frequency_hz=mount.get('frequency_hz'),
                           mount_damping=mount.get('damping'), resonant=mount.get('resonant'),
                           arrival_quality=c.get('arrival_quality'), stream_delay_s=c.get('stream_delay_s')))
    s['events'] = ev
    delays = [e['stream_delay_s'] for e in ev if e.get('stream_delay_s') is not None]
    s['stream_delay_s'] = dict(median=_median(delays), n=len(delays)) if delays else None
    s['resonant_at_events'] = any(e.get('resonant') for e in ev) if ev else None
    notes = []
    peaks = [p['frequency_hz'] for p in (s['ambient'] or {}).get('peaks') or []]
    for e in ev:
        f = e.get('mount_frequency_hz')
        if f and peaks:
            close = [g for g in peaks if near(f, g)]
            notes.append(f"{e['event_id']}: mount at {f:g} Hz, " +
                         (f"an ambient peak ({close[0]:g} Hz)" if close else 'not among the ambient peaks'))
    if not any(s.get(k) for k in ('day', 'twilight', 'night')) and with_stream:
        notes.append('no clip with known light')
    s['notes'] = notes
    return s


CSV_FIELDS = ['camera_id', 'name', 'lat', 'lon', 'clips', 'uptime', 'latest_clip_utc', 'width', 'height', 'fps',
              'bitrate_kbps', 'day_usable', 'day_problems', 'twilight_usable', 'twilight_problems', 'night_usable',
              'night_problems', 'touring', 'calm_jitter_px', 'vibration_usable', 'vibration_problems',
              'ambient_frequency_hz', 'ambient_peaks_hz', 'ambient_damping', 'ambient_rms_px', 'events',
              'resonant_at_events', 'mount_frequency_at_events_hz', 'stream_delay_s', 'notes']


def csv_row(s):
    st, amb = s.get('stream') or {}, s.get('ambient') or {}
    day, dusk, night = s.get('day') or {}, s.get('twilight') or {}, s.get('night') or {}
    return dict(camera_id=s['camera_id'], name=s['name'], lat=s['lat'], lon=s['lon'], clips=s['clips'],
                uptime=s.get('uptime'), latest_clip_utc=s.get('latest_clip_utc'),
                width=st.get('width'), height=st.get('height'), fps=st.get('fps'), bitrate_kbps=st.get('bitrate_kbps'),
                day_usable=day.get('usable'), day_problems='; '.join(day.get('problems') or []),
                twilight_usable=dusk.get('usable'), twilight_problems='; '.join(dusk.get('problems') or []),
                night_usable=night.get('usable'), night_problems='; '.join(night.get('problems') or []),
                touring=s['touring'], calm_jitter_px=s['calm_jitter_px'], vibration_usable=s['vibration_usable'],
                vibration_problems='; '.join(s['vibration_problems']), ambient_frequency_hz=amb.get('frequency_hz'),
                ambient_peaks_hz=' '.join(f"{p['frequency_hz']:g}" for p in amb.get('peaks') or []),
                ambient_damping=amb.get('damping'), ambient_rms_px=amb.get('rms_px'), events=len(s['events']),
                resonant_at_events=s['resonant_at_events'],
                mount_frequency_at_events_hz=' '.join(str(e['mount_frequency_hz']) for e in s['events']
                                                      if e.get('mount_frequency_hz')),
                stream_delay_s=(s['stream_delay_s'] or {}).get('median'), notes='; '.join(s['notes']))


def plot_camera(s, clips, path):
    """The ambient spectra of a camera's clips (by light), with the mount frequencies measured at events"""
    import matplotlib.pyplot as plt
    colors = dict(day=SERIES[0], twilight=SERIES[3], night=SERIES[1])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    shown = set()
    for c in sorted(clips, key=lambda c: c.get('start_utc') or ''):
        a = c.get('ambient')
        if not a:
            continue
        fr, p = np.array(a['spectrum']['frequency_hz']), np.array(a['spectrum']['psd_px2_per_hz'])
        keep = (fr >= 0.5) & (fr <= 0.4 * a['fps'])
        light = c.get('light', 'night')
        ax.semilogy(fr[keep], p[keep], color=colors.get(light, MUTED), lw=1.6, alpha=0.85,
                    label=None if light in shown else light)
        shown.add(light)
    for e in s['events']:
        if e.get('mount_frequency_hz'):
            ax.axvline(e['mount_frequency_hz'], color=INK, ls='--', lw=1.2)
            ax.text(e['mount_frequency_hz'], 0.98, f" event {e['event_id'][:8]}", transform=ax.get_xaxis_transform(),
                    va='top', fontsize=11, color=INK)
    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Ambient image motion (px²/Hz)', fontsize=13)
    amb = s.get('ambient') or {}
    sub = f"peak {amb['frequency_hz']:g} Hz" if amb.get('frequency_hz') else 'no clear peak'
    ax.set_title(f"{s['camera_id']} {s['name']}: ambient vibration, {sub}", fontsize=14, loc='left')
    style(ax)
    ax.tick_params(labelsize=12)
    if shown:
        ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_catalog(out, cameras=None):
    """The catalog from all the clips and events in out"""
    clips = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, 'clips', '*', '*.json')))]
    events = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, 'events', '*.json')))]
    known = {}
    path = os.path.join(out, 'cameras.json')
    if os.path.exists(path):
        known = {c['camera_id']: c for c in json.load(open(path))}
    for e in events:                                  # cameras seen only at events
        for cid, c in e['cameras'].items():
            known.setdefault(cid, dict(camera_id=cid, name=c.get('name', ''), lat=c.get('lat'), lon=c.get('lon'),
                                       view=c.get('view', '')))
    for c in cameras or []:
        known[c['camera_id']] = {k: c.get(k) for k in ('camera_id', 'name', 'lat', 'lon', 'view', 'video_url')}
    with open(path, 'w') as f:
        json.dump(list(known.values()), f, indent=1)
    os.makedirs(os.path.join(out, 'calibration'), exist_ok=True)
    before = {}
    if os.path.exists(os.path.join(out, 'catalog.json')):
        before = {c['camera_id']: c for c in json.load(open(os.path.join(out, 'catalog.json')))['cameras']}
    catalog = []
    for cid, cam in sorted(known.items()):
        mine = [c for c in clips if c['camera_id'] == cid]
        s = summarize(cam, mine, events)
        if mine:
            s['figure'] = f'calibration/{cid}.png'
            plot_camera(s, mine, os.path.join(out, s['figure']))
        catalog.append(s)
    with open(os.path.join(out, 'catalog.json'), 'w') as f:
        json.dump(dict(built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                       quality_bars=QUALITY, cameras=catalog), f, indent=1)
    with open(os.path.join(out, 'catalog.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_row(s) for s in catalog)
    changes = [c for s in catalog for c in changes_of(before.get(s['camera_id']), s)]
    with open(os.path.join(out, 'changes.json'), 'w') as f:
        json.dump(dict(built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), changes=changes),
                  f, indent=1)
    return catalog


def changes_of(old, new):
    """What changed in a camera's entry since the catalog before: new, offline or back, usable or not by day,
    by night and for vibration, a new strongest ambient peak"""
    cid = f"{new['camera_id']} {new['name']}"
    if old is None:
        return [f'{cid}: new in the catalog']
    out = []
    if 'offline' in (new.get('latest_problems') or []) and 'offline' not in (old.get('latest_problems') or []):
        out.append(f'{cid}: offline at the latest clip')
    if 'offline' in (old.get('latest_problems') or []) and new.get('latest_problems') is not None and \
            'offline' not in new['latest_problems']:
        out.append(f'{cid}: back online')
    for light in ('day', 'twilight', 'night'):
        a, b = (old.get(light) or {}).get('usable'), (new.get(light) or {}).get('usable')
        if a is not None and b is not None and a != b:
            why = '; '.join((new.get(light) or {}).get('problems') or [])
            out.append(f"{cid}: {'now usable' if b else 'no longer usable'} by {light}" + (f' ({why})' if why else ''))
    a, b = old.get('vibration_usable'), new.get('vibration_usable')
    if a is not None and b is not None and a != b:
        out.append(f"{cid}: {'now usable' if b else 'no longer usable'} for vibration")
    fa, fb = (old.get('ambient') or {}).get('frequency_hz'), (new.get('ambient') or {}).get('frequency_hz')
    if fa and fb and not near(fa, fb):
        out.append(f'{cid}: strongest ambient peak {fa:g} -> {fb:g} Hz')
    if old.get('touring') is False and new.get('touring'):
        out.append(f'{cid}: now touring views')
    return out


def clip_id_of(row):
    if row.get('clip_id'):
        return row['clip_id']
    t = parse_time(row.get('start_utc'))
    base = t.strftime('%Y%m%dT%H%M%S') if t else os.path.splitext(os.path.basename(row.get('video', 'clip')))[0]
    return base + (f"_{float(row['start_s']):.0f}s" if row.get('start_s') else '')


def prune(recordings, keep_days):
    """Delete the videos of the recordings older than keep_days (their clips.csv stays)"""
    limit = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)
    for folder in glob.glob(os.path.join(recordings, '*')):
        try:
            t = dt.datetime.strptime(os.path.basename(folder), '%Y%m%dT%H%M%SZ').replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if t < limit:
            for video in glob.glob(os.path.join(folder, '*.mp4')):
                os.remove(video)


def calibrate(clips_csv, videos, out, events=(), only=None, streams=None, record_s=180.0, wind=False,
              keep_days=7.0, workers=4):
    """Calibrate the clips listed (camera_id, name, lat, lon, video, and optionally start_utc, view, clip_id,
    start_s, end_s, wind_mps, error), add the events' runs, and rebuild the catalog. With streams, first record
    a clip of each stream into out/recordings/<UTC time>/ (and delete the videos older than keep_days)"""
    if streams:
        from .record import record
        videos, clips_csv = record(streams, os.path.join(out, 'recordings'), record_s, only, wind, workers)
    rows = read_cameras(clips_csv) if clips_csv else []
    if only:
        rows = [r for r in rows if r['camera_id'] in only]
    for row in rows:
        row['clip_id'] = clip_id_of(row)
        cdir = os.path.join(out, 'clips', row['camera_id'])
        os.makedirs(cdir, exist_ok=True)
        try:
            r = calibrate_clip(row, videos)
        except (subprocess.CalledProcessError, ValueError) as e:
            r = dict(camera_id=row['camera_id'], clip_id=row['clip_id'], problems=[f'failed: {e}'],
                     vibration_problems=[])
        with open(os.path.join(cdir, f"{row['clip_id']}.json"), 'w') as f:
            json.dump(r, f, indent=1)
        amb = r.get('ambient') or {}
        if 'offline' in r['problems']:
            print(f"{row['camera_id']} {row['clip_id']}: offline", flush=True)
            continue
        print(f"{row['camera_id']} {row['clip_id']}: {r.get('light', '-')}, "
              f"{'; '.join(r['problems']) or 'usable'}; vibration: {'; '.join(r['vibration_problems']) or 'usable'}; "
              f"ambient {amb.get('frequency_hz', 'no')} Hz peak", flush=True)
    os.makedirs(os.path.join(out, 'events'), exist_ok=True)
    for run in events:
        e = event_record(run)
        with open(os.path.join(out, 'events', f"{e['event_id']}.json"), 'w') as f:
            json.dump(e, f, indent=1)
    catalog = build_catalog(out, rows)
    if streams and keep_days is not None:
        prune(os.path.join(out, 'recordings'), keep_days)
    print(f'{len(catalog)} cameras in {os.path.join(out, "catalog.json")}')
    for line in json.load(open(os.path.join(out, 'changes.json')))['changes']:
        print('  ' + line)
    return catalog
