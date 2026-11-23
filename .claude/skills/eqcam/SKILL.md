---
name: eqcam
description: Rapid survey of camera videos after an earthquake - prepare the frames before, during and after the shaking, review them for structural damage, landslides, ground failure, fire, smoke, outages and shaking intensity (with Claude, Gemini or Qwen3-VL through an API, or in this session), and make the table and map. Use when given an event's camera videos (e.g., downloaded by the ShakeAlert/PDL server) or asked to survey or review earthquake camera footage. Also the periodic survey and calibration of the cameras outside events (record their live streams, check quality day and night, measure the mount's resonance, update the camera catalog).
---

# eqcam: camera survey after an earthquake

The code is the `eqcam` package in this repository. If the `eqcam` command is not installed, run it from the
source: `PYTHONPATH=<repo>/src python -m eqcam.cli ...` (needs numpy, scipy, matplotlib, pillow, ffmpeg and
ffprobe; the API backends also need `pip install "eqcam[claude]"` or `"eqcam[gemini]"`).

## 0. Inputs

- **videos.csv**: one row per camera: `camera_id, name, lat, lon, video` (path relative to the videos folder),
  and optionally `start_utc` (ISO time of the first frame), `view` (what the camera looks at) and `video_url`
  (the video online, linked from the map). With `start_utc` and the event, the shaking is searched near the
  predicted arrivals.
- **event.json**: `event_id, origin_utc, lat, lon, depth_km, magnitude, place` (only `event_id` required), and
  optionally `magnitude_type`, `note` and `url` (the event page, e.g., USGS). The alert's first magnitude is often
  revised: when the event is in a catalog, set `url` and use the catalog's magnitude (check it again before
  republishing).
- If the server's list has another layout, write a small conversion to this CSV. Never modify the videos.

## 1. Prepare

```bash
eqcam prepare --cameras videos.csv --videos <folder> --event event.json --out runs/<event_id>
```

It writes, per camera, `cameras/<id>/tile.jpg` (BEFORE, PEAK SHAKING, AFTER, LATEST or LATE SHAKING, labeled),
the full frames, `motion.png`, and `metrics.json` and `tasks.json` (high priority first). Report the counts by
priority and the flags (no_video, dark_at_the_end, frozen_at_the_end, camera_moved, scene_changed,
view_changes, latest_in_another_view, shaking_until_a_view_change, resonant_mount). A `resonant_mount` camera
rings at its mount's own frequency (a pole, a mast, the building): its image motion overstates the ground
shaking, so do not compare image motion across cameras as intensity. Cameras that pan or tour their preset views
are split into stable views: the shaking is searched only within a view (the image jittering back and forth, not
drifting), and BEFORE, PEAK and AFTER are in that view; `motion_text` lists the view changes and any sudden
change of the scene in the same view.

## 2. Review

Choose with the user:

**a. An API backend** (the operational path). These calls cost money: before a first run, say how many
cameras and roughly how many tokens (about 1,200 per tile plus about 1,000 of text and 300-1,000 of output per
camera) and get the user's go-ahead. Frames showing people go to the provider: check that the user is fine
with that for this provider.

```bash
eqcam review --out runs/<id> --backend claude                                  # claude-opus-5-5
eqcam review --out runs/<id> --backend claude --model claude-haiku-5-5 --stage screen
eqcam review --out runs/<id> --backend claude --stage final --escalate-from screen   # flagged cameras only
eqcam review --out runs/<id> --backend claude --platform vertex --project <gcp-project>
eqcam review --out runs/<id> --backend gemini --project <gcp-project>
eqcam review --out runs/<id> --backend qwen --base-url http://<gpu-host>:8000/v1
```

**b. In this session** (no API key, testing, or a few cameras). Read the instructions once:
`SYSTEM_PROMPT` and `FINDINGS_SCHEMA` in `src/eqcam/schema.py`. Then for each task in `tasks.json` with a tile,
high priority first:

1. Read `cameras/<id>/tile.jpg`, and the task's `motion_text`. Read the full frames (`frames`) when something
   is small or ambiguous, and `motion.png` when the timing matters.
2. Write `runs/<id>/findings/review/<camera_id>.json`:
   ```json
   {"camera_id": "...", "stage": "review", "reviewed_by": "claude-code", "model": "<your model id>",
    "reviewed_at": "<UTC ISO time>", "latency_s": null, "usage": {}, "error": null,
    "findings": { ...FINDINGS_SCHEMA... }}
   ```
   Add `"video_window_s": [start, end]` (s of the video) when you located what you saw (the shaking or an
   effect) more precisely than `motion_text`; the map links the video at that window.
3. Hold to the instructions: compare BEFORE with AFTER and LATEST; report only what is visible; "uncertain"
   rather than a guess; common false alarms (fog, low sun, lens flare, rain on the lens, steam, traffic dust,
   motion blur, a camera that panned); intensity only from visible effects, by the MMI rubric of the
   instructions (`mmi_kind`: an "estimate" from effects seen; an "upper bound" when the scene could show effects
   but shows none; else "not constrained"), never from the image motion; people only as counts and actions.
4. Fill `people_boxes`: a box around every person or close group in any of the four frames, in 0-1000 of the
   whole tile (x from the left, y from the top), close around each person or group (eqcam adds a 10 % margin;
   do not box whole sidewalks, which blurs the scene); include small, distant and partly hidden people
   (look at the full frames when the tile is too small to tell); `[]` if there are none. `eqcam report` blurs
   them in the tile and the frames before anything is published; a missed person stays visible, an extra box
   only blurs a little more.

With more than about 15 cameras, split them among parallel subagents: give each a list of camera ids, the
path of the run, and steps 1-4; each writes its own files. Then check them all:

```bash
eqcam validate --out runs/<id> --stage review
```

and fix every problem it lists.

## 3. Report

```bash
eqcam report --out runs/<id>
```

It first blurs the people in the tiles and frames from the `people_boxes` of the review (originals kept in
`cameras/<id>/original/`; `--no-mask` to skip), then writes `survey.csv`, `survey.json`, `map.html` (open it
in a browser; markers colored by severity, a panel to filter the cameras by tag, and in each popup the tile and
links to the video online, `video_url`, at the window of the detections) and `map.png`. Tell the user:

- the counts by severity, and every camera of severity 2 or 3: what was detected, where, the confidence, and
  the path of its tile;
- the cameras that went dark or froze (possible power or network outages) and those without video;
- that the findings are automated and unverified: a person should look at the tiles before anyone acts on them.

## 4. Vibration (optional, for scientific use)

Only when the user asks for the vibration, frequencies or displacements of the cameras or of what they see (not
for the rapid survey: it decodes the videos again, about 5 s per camera):

```bash
eqcam report --out runs/<id> --vibration
```

It writes `vibration.html` (the cameras colored by mount resonance; in each popup the figure), and
`vibration_summary.png` (the dominant frequencies against the distance; the spectra of the cameras), and per
camera `cameras/<id>/vibration.png` (the frame with the vibrating objects outlined; the camera's displacement
against time, spectrum and spectrogram; the objects' displacement relative to the camera and spectra),
`vibration.json` (the numbers) and `vibration.npz` (the series); `map.html` links them. Tell the user, per camera,
the dominant frequency, whether the mount resonates, the roll, and the objects found (where, at which frequency);
that displacements are in pixels (they depend on the zoom, the distance and the mount: compare frequencies and
shapes, not amplitudes); and look at a few `vibration.png` before relying on the objects: a patch that a car,
a person or a light change crosses can still pass the checks.

The same step picks the P and S arrivals in each camera's motion: `arrivals.html` (the cameras by the quality
of their picks, "P and S", "S only" or "ambiguous"; per camera `arrivals.png` with the picks and the spectra of
the noise, P and S windows; `arrivals_summary.png`, S-P against the distance and the stream delays) and the data
for further analysis in `vibration_data/` and `vibration_data.zip` (motion and spectra CSV per camera,
`cameras.json`, a README). Tell the user the picks, S-P against the expected, and the stream delays (each pick
minus its predicted arrival; P and S must agree); P is often too weak near the source.

## 5. Camera survey and calibration (periodic, outside events)

When asked to survey, check or calibrate the cameras, or to update the calibration. The calibration lives in
one folder (below, `calibration/`; use the one the user names, ask if none is known; keep it out of git, the
recordings are large):

```
calibration/
  streams.csv               the camera list: camera_id, name, lat, lon, stream (a YouTube watch/live URL, or any
                            URL ffmpeg reads: HLS .m3u8, RTSP, HTTP), view
  catalog/
    catalog.csv, .json      one entry per camera (the current state; rebuilt every run)
    changes.json            what changed in the last run
    calibration/<id>.png    the ambient spectra of the camera's clips, with its mount frequencies at events
    clips/<id>/<clip>.json  the results of every clip (the history: keep)
    events/<event>.json     what each event measured per camera (keep)
    recordings/<UTC time>/  the clips recorded and their clips.csv (videos deleted after --keep-days)
```

**a. The camera list.** Add or update cameras in `streams.csv` when the user gives them: one row
per camera, with its location (it sets day or night and the distances at events). Keep `camera_id` stable: the
history and the events are matched by it, and at an event the `videos.csv` should use the same ids. Never
remove a row without asking: mark a retired camera in `view` instead.

**b. One survey run** (records every stream, calibrates the clips, adds to the catalog, prunes old videos):

```bash
eqcam calibrate --streams calibration/streams.csv --record 180 --wind --out calibration/catalog
```

YouTube streams need yt-dlp (`pip install "eqcam[record]"`, and `yt-dlp` on the PATH; keep it updated, YouTube
changes often). A stream that cannot be recorded counts as offline (the camera's `uptime`). It takes about
`--record` plus 15 s per camera, `--workers` (4) streams at a time.

**c. Periodically.** The calibration needs clips by day, twilight and night, and on calm and windy days, so run
it at several times of day, e.g., every 6 h; on wind advisory days add runs (`--wind` records the wind at the
nearest weather station, api.weather.gov, US only). Unattended, on the machine that records the streams, with
a scheduler such as cron:

```
15 */6 * * *  eqcam calibrate --streams <dir>/streams.csv --record 180 --wind --out <dir>/catalog >> <dir>/calibrate.log 2>&1
```

If that machine already records the streams on a schedule, keep its recorder and pass each batch of files
instead: `--clips clips.csv --videos <folder>` (`camera_id, name, lat, lon, video, start_utc`; the same
`camera_id` as at events). Ask the user before installing a scheduled job. In a session,
`/loop 6h /eqcam survey the cameras` does the same while the session is open. When the user brings back a
catalog folder from another machine, read it as in d.

**d. After each run, tell the user** from `changes.json` what changed (cameras new, offline or back online, no
longer usable by day or night or for vibration, a new strongest ambient peak, now touring), and from
`catalog.csv` the counts: cameras usable by day, by night, for vibration; offline at the latest run; with an
ambient peak repeated over several clips. Say nothing more when nothing changed.

**e. After each event**, add what it measured (the mount in the shaking, and the stream delays with
`report --vibration`):

```bash
eqcam calibrate --events runs/<event_id>/run --out calibration/catalog
```

Per camera, the catalog gives: usable by day, twilight and night (and the problems: low resolution, dark,
frozen, blurred), fixed or touring, usable for vibration (frame rate, a stable view, steady), the ambient peaks
(grouped over the clips; a peak in many clips is the mount) and whether the mount frequency measured at events
is among them, the uptime and the median stream delay. The quality bars (`QUALITY` in `calibrate.py`) are
provisional: do not change them without the user. One short clip says little about the mount: look at
`calibration/<id>.png` before relying on a frequency. Older clips can also be calibrated from files with
`--clips clips.csv --videos <folder>` (the columns of `videos.csv`, and optionally `clip_id`, `start_s`,
`end_s`, `wind_mps`).
