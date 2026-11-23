---
name: eqcam
description: Rapid survey of camera videos after an earthquake - prepare the frames before, during and after the shaking, review them for structural damage, landslides, ground failure, fire, smoke, outages and shaking intensity (with Claude, Gemini or Qwen3-VL through an API, or in this session), and make the table and map. Use when given an event's camera videos (e.g., downloaded by the ShakeAlert/PDL server) or asked to survey or review earthquake camera footage.
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
- **event.json**: `event_id, origin_utc, lat, lon, depth_km, magnitude, place` (only `event_id` required).
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
