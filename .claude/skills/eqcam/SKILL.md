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
  and optionally `start_utc` (ISO time of the first frame) and `view` (what the camera looks at). With
  `start_utc` and the event, the shaking is searched near the predicted arrivals.
- **event.json**: `event_id, origin_utc, lat, lon, depth_km, magnitude, place` (only `event_id` required).
- If the server's list has another layout, write a small conversion to this CSV. Never modify the videos.

## 1. Prepare

```bash
eqcam prepare --cameras videos.csv --videos <folder> --event event.json --out runs/<event_id>
```

It writes, per camera, `cameras/<id>/tile.jpg` (BEFORE, PEAK SHAKING, AFTER, LATEST or LATE SHAKING, labeled),
the full frames, `motion.png`, and `metrics.json` and `tasks.json` (high priority first). Report the counts by
priority and the flags (no_video, dark_at_the_end, frozen_at_the_end, camera_moved, scene_changed, scene_cuts).

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
3. Hold to the instructions: compare BEFORE with AFTER and LATEST; report only what is visible; "uncertain"
   rather than a guess; common false alarms (fog, low sun, lens flare, rain on the lens, steam, traffic dust,
   motion blur, a camera that panned); intensity only from visible effects; people only as counts and actions.

With more than about 15 cameras, split them among parallel subagents: give each a list of camera ids, the
path of the run, and steps 1-3; each writes its own files. Then check them all:

```bash
eqcam validate --out runs/<id> --stage review
```

and fix every problem it lists.

## 3. Report

```bash
eqcam report --out runs/<id>
```

It writes `survey.csv`, `survey.json`, `map.html` (open it in a browser; markers colored by severity, the tile in
each popup) and `map.png`. Tell the user:

- the counts by severity, and every camera of severity 2 or 3: what was detected, where, the confidence, and
  the path of its tile;
- the cameras that went dark or froze (possible power or network outages) and those without video;
- that the findings are automated and unverified: a person should look at the tiles before anyone acts on them.
