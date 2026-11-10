"""review: the findings of each camera from a vision-language backend, in parallel

Each record is written to OUT/findings/<stage>/<camera_id>.json:
  {camera_id, stage, reviewed_by, model, reviewed_at, latency_s, usage, error, findings}
where findings follows eqcam.schema.FINDINGS_SCHEMA. A person or a Claude Code session can write the same
records by hand (reviewed_by "claude-code"); eqcam validate checks them.

A second stage reviews only the cameras the first one flagged (escalate_from), e.g., a cheap model on every
camera, then a stronger one on those of severity 2 or more or of low confidence.
"""

import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import user_text, validate


def findings_path(out, stage, camera_id):
    return os.path.join(out, 'findings', stage, f'{camera_id}.json')


def load_record(out, stage, camera_id):
    path = findings_path(out, stage, camera_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def flagged(record, min_severity=2):
    """Whether a reviewed camera needs a second look: severity at least min_severity, low confidence, or no
    usable findings"""
    if record is None:
        return False
    fd = record.get('findings')
    if not fd or record.get('error'):
        return True
    return fd.get('severity', 0) >= min_severity or fd.get('confidence') == 'low'


def select_tasks(out, stage, only=None, escalate_from=None, min_severity=2, overwrite=False):
    with open(os.path.join(out, 'tasks.json')) as f:
        tasks = json.load(f)
    tasks = [t for t in tasks if t.get('tile')]                  # no video, nothing to review
    if only:
        tasks = [t for t in tasks if t['camera_id'] in only]
    if escalate_from:
        tasks = [t for t in tasks if flagged(load_record(out, escalate_from, t['camera_id']), min_severity)]
    if not overwrite:
        tasks = [t for t in tasks if not os.path.exists(findings_path(out, stage, t['camera_id']))]
    return tasks


def review_one(backend, out, stage, task):
    with open(os.path.join(out, task['tile']), 'rb') as f:
        image = f.read()
    t0 = time.monotonic()
    result = backend.review(image, user_text(task))
    record = dict(camera_id=task['camera_id'], stage=stage, reviewed_by=backend.name, model=result.model,
                  reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                  latency_s=round(time.monotonic() - t0, 2), usage=result.usage, error=result.error,
                  findings=result.findings)
    if result.error and result.raw:
        record['raw'] = result.raw[:4000]
    path = findings_path(out, stage, task['camera_id'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(record, f, indent=1)
    return record


def review(out, backend, stage='review', workers=8, only=None, escalate_from=None, min_severity=2,
           overwrite=False):
    """Review the cameras of OUT/tasks.json (high priority first) with backend; returns the records"""
    tasks = select_tasks(out, stage, only, escalate_from, min_severity, overwrite)
    print(f'{len(tasks)} cameras to review with {backend.name} ({getattr(backend, "model", "")}), stage {stage}')
    records, t0 = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(review_one, backend, out, stage, t): t for t in tasks}
        for k, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                r = future.result()
            except Exception as err:                          # noqa: BLE001 - one camera must not stop the rest
                r = dict(camera_id=task['camera_id'], stage=stage, error=f'{type(err).__name__}: {err}',
                         findings=None)
            records.append(r)
            fd = r.get('findings') or {}
            status = f"error: {r['error']}" if r.get('error') else f"severity {fd.get('severity')}: {fd.get('summary')}"
            print(f"[{k}/{len(tasks)}] {r['camera_id']}: {status}")
    tokens_in = sum((r.get('usage') or {}).get('input_tokens') or 0 for r in records)
    tokens_out = sum((r.get('usage') or {}).get('output_tokens') or 0 for r in records)
    print(f'{len(records)} reviewed in {time.monotonic() - t0:.1f} s; {tokens_in} input and {tokens_out} output tokens')
    return records


def validate_stage(out, stage):
    """The problems of every record of a stage (for records written by hand or in a Claude Code session)"""
    folder = os.path.join(out, 'findings', stage)
    problems = {}
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not name.endswith('.json'):
            continue
        with open(os.path.join(folder, name)) as f:
            try:
                record = json.load(f)
            except json.JSONDecodeError as err:
                problems[name] = [f'not JSON: {err}']
                continue
        issues = [] if record.get('camera_id') == name[:-5] else ['camera_id does not match the file name']
        issues += validate(record.get('findings'))
        if issues:
            problems[name] = issues
    return problems
