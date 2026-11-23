"""record: clips of the cameras' live streams, for calibrate

streams.csv: camera_id, name, lat, lon, stream (a YouTube watch or live URL, read through yt-dlp; or any URL
ffmpeg reads: HLS .m3u8, RTSP, HTTP), and optionally view. Each run writes <out>/<UTC time>/<camera_id>.mp4
(video only, copied as streamed) and clips.csv (the camera, the video, start_utc, the error if the stream could
not be recorded, and with wind the wind speed at the nearest weather station reporting it), ready for calibrate.
"""

import csv
import datetime as dt
import json
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .frames import probe, read_cameras

USER_AGENT = 'eqcam (research; camera calibration)'


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def is_youtube(url):
    return any(h in url for h in ('youtube.com/', 'youtu.be/'))


def source_of(url, timeout=60):
    """The URL ffmpeg reads: for YouTube, the HLS stream of the best video up to 1080p (live streams come as
    video-only formats), through yt-dlp"""
    if not is_youtube(url):
        return url
    out = subprocess.run(['yt-dlp', '-q', '--no-warnings', '-f', 'bestvideo[height<=1080]/best[height<=1080]/best',
                          '-g', url], capture_output=True, text=True, timeout=timeout)
    if out.returncode or not out.stdout.strip():
        raise RuntimeError((out.stderr.strip().splitlines() or ['yt-dlp failed'])[-1][:200])
    return out.stdout.strip().splitlines()[0]


def record_one(row, folder, seconds):
    """Record seconds of one camera's stream; the row of clips.csv"""
    cid = row['camera_id']
    name = f'{cid}.mp4'
    path = os.path.join(folder, name)
    clip = {k: row.get(k, '') for k in ('camera_id', 'name', 'lat', 'lon', 'view')}
    clip.update(video=name, start_utc=now_utc(), error='')          # the time of the attempt, if it fails
    try:
        src = source_of(row['stream'])
        clip['start_utc'] = now_utc()
        out = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-t', f'{seconds:.0f}',
                              '-map', '0:v:0', '-c', 'copy', path], capture_output=True, text=True,
                             timeout=seconds + 120)
        if out.returncode or not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError((out.stderr.strip().splitlines() or ['ffmpeg failed'])[-1][:200])
        duration = probe(path)[2]
        if not duration == duration or duration < min(10.0, 0.5 * seconds):
            raise RuntimeError(f'only {duration:.0f} s recorded')
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError, KeyError) as e:
        clip['error'] = str(e) or type(e).__name__
    return clip


def _get_json(url, timeout=20):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/geo+json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wind_at(lat, lon, stations=5, max_age_h=2.0):
    """The latest wind speed (m/s) and its station: the nearest of the first few weather stations to lat, lon
    that reported one within max_age_h (US: api.weather.gov); (None, '') if none"""
    try:
        point = _get_json(f'https://api.weather.gov/points/{lat:.4f},{lon:.4f}')
        features = _get_json(point['properties']['observationStations'])['features'][:stations]
    except (OSError, KeyError, ValueError, TypeError):
        return None, ''
    now = dt.datetime.now(dt.timezone.utc)
    for feature in features:
        sid = feature['properties']['stationIdentifier']
        try:
            obs = _get_json(f'https://api.weather.gov/stations/{sid}/observations/latest')['properties']
            speed = obs['windSpeed']['value']                           # km/h
            age = (now - dt.datetime.fromisoformat(obs['timestamp'])).total_seconds() / 3600
        except (OSError, KeyError, ValueError, TypeError):
            continue
        if speed is not None and age <= max_age_h:
            return round(speed / 3.6, 1), sid
    return None, ''


FIELDS = ['camera_id', 'name', 'lat', 'lon', 'video', 'start_utc', 'view', 'wind_mps', 'wind_station', 'error']


def record(streams_csv, out, seconds=180.0, only=None, wind=False, workers=4):
    """Record a clip of every stream into <out>/<UTC time>/; the folder and its clips.csv"""
    rows = [r for r in read_cameras(streams_csv) if r.get('stream') and (not only or r['camera_id'] in only)]
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = os.path.join(out, stamp)
    os.makedirs(folder, exist_ok=True)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        clips = list(pool.map(lambda r: record_one(r, folder, seconds), rows))
    for clip, row in zip(clips, rows):
        speed, station = wind_at(row['lat'], row['lon']) if wind and row.get('lat') is not None else (None, '')
        clip.update(wind_mps='' if speed is None else speed, wind_station=station)
        print(f"{clip['camera_id']}: {'recorded' if not clip['error'] else 'failed: ' + clip['error']}", flush=True)
    path = os.path.join(folder, 'clips.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(clips)
    print(f"{sum(not c['error'] for c in clips)} of {len(clips)} streams recorded in {folder}")
    return folder, path
