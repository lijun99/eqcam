"""report: the motion and the findings of every camera in a table (survey.csv, survey.json) and maps
(map.html with Leaflet, map.png); for each camera the findings of the first stage in stages that has them"""

import csv
import html
import json
import math
import os

import matplotlib
import numpy as np

from .review import load_record
from .schema import DETECTIONS, SEVERITY_NAMES

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

COLORS = {0: '#8a8f98', 1: '#e3b505', 2: '#f08a24', 3: '#d62728', None: '#4a6fa5'}
NAMES = {**SEVERITY_NAMES, None: 'not reviewed'}


def merge(out, stages=('final', 'review', 'screen')):
    with open(os.path.join(out, 'metrics.json')) as f:
        metrics = json.load(f)
    rows = []
    for m in metrics:
        record = next((r for s in stages if (r := load_record(out, s, m['camera_id'])) is not None), None)
        fd = (record or {}).get('findings') or {}
        rows.append(dict(m, record=record, findings=fd, severity=fd.get('severity')))
    return rows


def detected(fd):
    """The detections present or uncertain, as 'name (yes)' strings"""
    return [f"{key} ({fd[key]['present']})" for key in DETECTIONS
            if isinstance(fd.get(key), dict) and fd[key].get('present') in ('yes', 'uncertain')]


def write_table(rows, event, out):
    fields = ['camera_id', 'name', 'lat', 'lon', 'distance_km', 'priority', 'flags', 'shaking_detected', 'onset_s',
              'strong_duration_s', 'shake_amplitude_px', 'peak_image_velocity_px_s', 'permanent_shift_px',
              'changed_fraction_after', 'changed_fraction_latest', 'reviewed_by', 'model', 'review_error',
              'severity', 'confidence', 'camera_status', 'shaking_visible', 'intensity_mmi', *DETECTIONS, 'summary']
    with open(os.path.join(out, 'survey.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            fd, rec = r['findings'], r['record'] or {}
            vals = dict(r, flags='; '.join(r.get('flags', [])), reviewed_by=rec.get('reviewed_by'),
                        model=rec.get('model'), review_error=rec.get('error'), confidence=fd.get('confidence'),
                        camera_status=fd.get('camera_status'), summary=fd.get('summary'),
                        shaking_visible=(fd.get('shaking') or {}).get('visible'),
                        intensity_mmi=(fd.get('shaking') or {}).get('intensity_mmi'),
                        **{k: (fd.get(k) or {}).get('present') for k in DETECTIONS})
            w.writerow(['' if vals.get(k) is None else vals[k] for k in fields])
    with open(os.path.join(out, 'survey.json'), 'w') as f:
        json.dump(dict(event=event, cameras=rows), f, indent=1)


def popup(r):
    e = html.escape
    fd, rec = r['findings'], r['record'] or {}
    lines = [f"<b>{e(r['camera_id'])}</b> {e(r.get('name') or '')}"]
    if fd:
        lines.append(f"<i>{e(NAMES.get(fd.get('severity')))}</i> ({e(str(fd.get('confidence')))} confidence, "
                     f"{e(str(rec.get('reviewed_by')))} {e(str(rec.get('model') or ''))}): {e(fd.get('summary', ''))}")
        for key in DETECTIONS:
            item = fd.get(key) or {}
            if item.get('present') in ('yes', 'uncertain'):
                lines.append(f"{e(key.replace('_', ' '))}: {e(item['present'])}. {e(item.get('description', ''))} "
                             f"{e(item.get('where', ''))}")
        shaking = fd.get('shaking') or {}
        if shaking.get('intensity_mmi'):
            lines.append(f"Intensity (MMI) {e(shaking['intensity_mmi'])}: {e(shaking.get('basis', ''))}")
    elif rec.get('error'):
        lines.append(f"<i>review failed: {e(rec['error'])}</i>")
    else:
        lines.append('<i>not reviewed</i>')
    lines.append(f"<small>{e(r.get('motion_text', ''))}</small>")
    if r.get('tile'):
        lines.append(f"<a href=\"{e(r['tile'])}\" target=\"_blank\"><img src=\"{e(r['tile'])}\" width=\"380\"></a>")
    return '<br>'.join(lines)


def title_of(event):
    return ' - '.join(str(event[k]) for k in ('event_id', 'place') if event.get(k)) or 'camera survey'


def map_html(rows, event, out):
    points = [dict(lat=r['lat'], lon=r['lon'], color=COLORS.get(r['severity'], COLORS[None]), label=r['camera_id'],
                   popup=popup(r)) for r in rows if r.get('lat') is not None]
    epi = [event['lat'], event['lon']] if event.get('lat') is not None else None
    title = html.escape(title_of(event))
    legend = ''.join(f'<div><span style="background:{COLORS[k]}"></span>{html.escape(v)}</div>'
                     for k, v in NAMES.items())
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .legend{{background:#fff;padding:6px 8px;font:12px sans-serif}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px}}</style>
</head><body><div id="map"></div><script>
const points = {json.dumps(points)};
const epi = {json.dumps(epi)};
const map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
            {{maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const bounds = [];
for (const p of points) {{
  L.circleMarker([p.lat, p.lon], {{radius: 8, color: '#222', weight: 1, fillColor: p.color, fillOpacity: 0.9}})
    .bindPopup(p.popup, {{maxWidth: 420}}).bindTooltip(p.label).addTo(map);
  bounds.push([p.lat, p.lon]);
}}
if (epi) {{ L.marker(epi).bindPopup({json.dumps(title)}).addTo(map); bounds.push(epi); }}
if (bounds.length > 1) map.fitBounds(bounds, {{padding: [40, 40]}}); else map.setView(bounds[0] || [34, -118], 10);
const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{ const d = L.DomUtil.create('div', 'legend'); d.innerHTML = {json.dumps(legend)}; return d; }};
legend.addTo(map);
</script></body></html>
"""
    with open(os.path.join(out, 'map.html'), 'w') as f:
        f.write(page)


def map_png(rows, event, out):
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    pts = [r for r in rows if r.get('lat') is not None]
    lats = [r['lat'] for r in pts] + ([event['lat']] if event.get('lat') is not None else [])
    for r in sorted(pts, key=lambda r: -1 if r['severity'] is None else r['severity']):
        ax.scatter(r['lon'], r['lat'], s=70, c=COLORS.get(r['severity'], COLORS[None]), edgecolors='#222',
                   linewidths=0.6, zorder=3)
        if (r['severity'] or 0) >= 2:
            ax.annotate(f"{r['camera_id']}: {', '.join(detected(r['findings']))[:70]}", (r['lon'], r['lat']),
                        xytext=(6, 4), textcoords='offset points', fontsize=7)
    if event.get('lat') is not None:
        ax.scatter(event['lon'], event['lat'], marker='*', s=260, c='k', zorder=4)
        ax.annotate(f"M{event.get('magnitude', '')}", (event['lon'], event['lat']), xytext=(8, -12),
                    textcoords='offset points', fontsize=9)
    if lats:
        ax.set_aspect(1 / math.cos(math.radians(float(np.mean(lats)))))
    for k, v in NAMES.items():
        ax.scatter([], [], c=COLORS[k], edgecolors='#222', linewidths=0.6, label=v)
    ax.legend(fontsize=7, loc='best', frameon=True)
    ax.set_xlabel('longitude', fontsize=9)
    ax.set_ylabel('latitude', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(title_of(event), fontsize=10, loc='left')
    fig.savefig(os.path.join(out, 'map.png'), dpi=130)
    plt.close(fig)


def report(out, stages=('final', 'review', 'screen')):
    with open(os.path.join(out, 'event.json')) as f:
        event = json.load(f)
    rows = merge(out, stages)
    write_table(rows, event, out)
    map_html(rows, event, out)
    map_png(rows, event, out)
    counts = {NAMES[k]: sum(1 for r in rows if r['severity'] == k) for k in NAMES}
    print(f'{len(rows)} cameras: ' + ', '.join(f'{v} {k}' for k, v in counts.items() if v))
    for r in sorted(rows, key=lambda r: -(r['severity'] or -1)):
        if (r['severity'] or 0) >= 2:
            print(f"  severity {r['severity']} {r['camera_id']}: {r['findings'].get('summary')}")
    print(f'-> {out}/survey.csv, survey.json, map.html, map.png')
    return rows
