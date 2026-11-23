"""report: the motion and the findings of every camera in a table (survey.csv, survey.json) and maps
(map.html with Leaflet, map.png); for each camera the findings of the first stage in stages that has them"""

import csv
import html
import json
import math
import os

import matplotlib
import numpy as np

from .frames import motion_items
from .mask import mask_run
from .review import load_record
from .schema import DETECTIONS, SEVERITY_NAMES

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.ticker import MaxNLocator           # noqa: E402

COLORS = {0: '#8a8f98', 1: '#e3b505', 2: '#f08a24', 3: '#d62728', None: '#4a6fa5'}
NAMES = {**SEVERITY_NAMES, None: 'not reviewed'}

# the basemaps of the maps (after `const map = L.map(...)`), and the view fitted to `bounds` with the basemap
# chosen by where it is
BASEMAPS_JS = r"""// basemaps that serve a page opened as a local file (OpenStreetMap's own tile servers block it):
// The National Map (USGS, public domain, United States only) and CARTO (worldwide, OpenStreetMap data)
const usgs = 'https://basemap.nationalmap.gov/arcgis/rest/services/';
const nationalMap = 'Basemap: <a href="https://www.usgs.gov/programs/national-geospatial-program/national-map">USGS The National Map</a>';
const bases = {
  'USGS Topo': L.tileLayer(usgs + 'USGSTopo/MapServer/tile/{z}/{y}/{x}', {maxZoom: 16, attribution: nationalMap}),
  'USGS Imagery': L.tileLayer(usgs + 'USGSImageryTopo/MapServer/tile/{z}/{y}/{x}', {maxZoom: 16, attribution: nationalMap}),
  'CARTO Light (worldwide)': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    {maxZoom: 19, subdomains: 'abcd',
     attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'}),
};
L.control.layers(bases, null, {position: 'topright'}).addTo(map);
"""
# the panels of the maps (filter, legend, information): a header (.ph) that folds the body (.pb); folded at first
# on phones (narrow or touch screens), so that they do not hide the map
PANEL_CSS = r""".ph{cursor:pointer;font-weight:bold;user-select:none} .ph::after{content:' \25BE'}
.closed .ph::after{content:' \25B8'} .closed .pb{display:none}
body.popup-open .filter, body.popup-open .legend, body.popup-open .box{display:none}"""
PANEL_JS = r"""const small = window.matchMedia('(max-width: 768px), (pointer: coarse)').matches;
function foldable(d, open) {
  d.querySelector('.ph').addEventListener('click', () => d.classList.toggle('closed'));
  if (!open) d.classList.add('closed');
  L.DomEvent.disableClickPropagation(d);
  L.DomEvent.disableScrollPropagation(d);
  return d;
}
// the popups: no taller than most of the screen (they scroll inside), the map panned to show them from their
// beginning, again when their images load or a pull-down opens; the panels hidden while one is open
const popupOptions = w => ({maxWidth: Math.min(w, window.innerWidth - 60),
  maxHeight: Math.round(window.innerHeight * 0.75), autoPanPadding: [12, 12]});
map.on('popupopen', e => {
  document.body.classList.add('popup-open');
  const el = e.popup.getElement();
  const top = () => { const c = el.querySelector('.leaflet-popup-content'); if (c) c.scrollTop = 0; };
  const again = () => { e.popup.update(); top(); };
  top();
  el.querySelectorAll('img').forEach(img => { if (!img.complete) img.addEventListener('load', again, {once: true}); });
  el.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => e.popup.update()));
});
map.on('popupclose', () => document.body.classList.remove('popup-open'));
"""
FIT_JS = r"""if (bounds.length > 1) map.fitBounds(bounds, {padding: [40, 40]}); else map.setView(bounds[0] || [34, -118], 10);
// The National Map where it has coverage (the conterminous US, Alaska, Hawaii, Puerto Rico), CARTO elsewhere
const c = map.getCenter();
const inUS = (c.lat > 24 && c.lat < 50 && c.lng > -125 && c.lng < -66) || (c.lat > 51 && c.lat < 72 && c.lng < -129)
  || (c.lat > 18 && c.lat < 23 && c.lng > -161 && c.lng < -154) || (c.lat > 17.5 && c.lat < 18.6 && c.lng > -68 && c.lng < -65);
bases[inUS ? 'USGS Topo' : 'CARTO Light (worldwide)'].addTo(map);
"""

# the tags the cameras can be filtered by on the map: the shaking, every detection, and camera outages
TAG_LABELS = {'shaking': 'Shaking', 'structural_damage': 'Structural damage',
              'infrastructure_damage': 'Road, bridge or utility damage', 'landslide': 'Landslide or rockfall',
              'ground_failure': 'Ground failure', 'fire': 'Fire', 'smoke_or_dust': 'Smoke or dust',
              'power_outage': 'Power outage', 'water': 'Flooding or water', 'people_response': 'People responding',
              'camera_problem': 'Camera dark, frozen or no video'}
TAGS = ['shaking', *DETECTIONS, 'camera_problem']
PROBLEM_FLAGS = ('no_video', 'dark_at_the_end', 'frozen_at_the_end')
PROBLEM_STATUS = ('dark', 'frozen', 'obstructed', 'unusable')


def tags(r):
    """The tags of a camera, {tag: 'yes' or 'uncertain'}"""
    fd, out = r['findings'], {}
    shaking = (fd.get('shaking') or {}).get('visible')
    if shaking in ('yes', 'uncertain'):
        out['shaking'] = shaking
    for key in DETECTIONS:
        present = (fd.get(key) or {}).get('present')
        if present in ('yes', 'uncertain'):
            out[key] = present
    if fd.get('camera_status') in PROBLEM_STATUS or any(f in PROBLEM_FLAGS for f in r.get('flags', [])):
        out['camera_problem'] = 'yes'
    return out


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
              'severity', 'confidence', 'camera_status', 'shaking_visible', 'intensity_mmi',
              'mmi_kind', 'mmi_confidence', *DETECTIONS, 'summary',
              'motion_brief', 'people_blurred', 'vibration_frequency_hz', 'vibration_objects',
              'arrival_quality', 'p_arrival_s', 's_arrival_s', 'stream_delay_s',
              'video_start_s', 'video_end_s', 'video_window', 'video_url']
    with open(os.path.join(out, 'survey.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            fd, rec = r['findings'], r['record'] or {}
            win = video_window(r) or (None, None, None)
            vib = r.get('vibration') or {}
            vals = dict(r, motion_brief='; '.join(items_of(r)),
                        vibration_frequency_hz=(vib.get('camera') or {}).get('frequency_hz'),
                        vibration_objects=len(vib['objects']) if vib else None,
                        arrival_quality=(r.get('arrivals') or {}).get('quality'),
                        p_arrival_s=(r.get('arrivals') or {}).get('p_s'),
                        s_arrival_s=(r.get('arrivals') or {}).get('s_s'),
                        stream_delay_s=(r.get('arrivals') or {}).get('delay_s_s'),
                        video_start_s=win[0], video_end_s=win[1], video_window=win[2],
                        flags='; '.join(r.get('flags', [])), reviewed_by=rec.get('reviewed_by'),
                        model=rec.get('model'), review_error=rec.get('error'), confidence=fd.get('confidence'),
                        camera_status=fd.get('camera_status'), summary=fd.get('summary'),
                        shaking_visible=(fd.get('shaking') or {}).get('visible'),
                        intensity_mmi=(fd.get('shaking') or {}).get('intensity_mmi'),
                        mmi_kind=(fd.get('shaking') or {}).get('mmi_kind'),
                        mmi_confidence=(fd.get('shaking') or {}).get('mmi_confidence'),
                        **{k: (fd.get(k) or {}).get('present') for k in DETECTIONS})
            w.writerow(['' if vals.get(k) is None else vals[k] for k in fields])
    with open(os.path.join(out, 'survey.json'), 'w') as f:
        json.dump(dict(event=event, cameras=rows), f, indent=1)


FRAME_LABELS = {'before': 'BEFORE', 'peak': 'PEAK', 'after': 'AFTER', 'latest': 'LATEST'}


def video_window(r):
    """The part of the video that covers the detections, (start, end, what) in s of the video: the window set by
    the reviewer (video_window_s in the record), else the camera vibration detected, else the predicted arrivals"""
    rec, dur = r.get('record') or {}, r.get('duration_s')
    clip = lambda a, b: (max(0.0, round(a, 1)), round(b if dur is None else min(b, dur), 1))  # noqa: E731
    w = rec.get('video_window_s')
    if isinstance(w, (list, tuple)) and len(w) == 2 and all(isinstance(x, (int, float)) for x in w) and w[1] > w[0]:
        return (*clip(*w), 'set by the review')
    if r.get('shaking_detected') and r.get('onset_s') is not None and r.get('end_s') is not None:
        return (*clip(r['onset_s'] - 5, r['end_s'] + 5), 'camera vibration detected')
    if r.get('predicted_p_s') is not None and r.get('predicted_s_s') is not None:
        return (*clip(r['predicted_p_s'] - 5, r['predicted_s_s'] + 30), 'predicted P and S arrivals')
    return None


def video_links(r):
    """Links to the video online (video_url; none without it), opened at the window of the detections and at
    the times of the frames (media fragments #t=start,end)"""
    e, src = html.escape, r.get('video_url')
    if not src:
        return ''
    w = video_window(r)
    frames = {k: v['video_time_s'] for k, v in (r.get('frames') or {}).items() if k in FRAME_LABELS}
    a = lambda frag, text: f'<a href="{e(src)}#t={frag}" target="_blank">{e(text)}</a>'  # noqa: E731
    parts = [a(f'{w[0]:g},{w[1]:g}', f'{w[0]:g}-{w[1]:g} s') + f' ({e(w[2])})'] if w else []
    parts += [a(f'{frames[k]:g}', f'{FRAME_LABELS[k]} {frames[k]:g} s') for k in FRAME_LABELS if k in frames]
    parts.append(f'<a href="{e(src)}" target="_blank">whole video</a>')
    return 'Video: ' + ' &middot; '.join(parts)


def items_of(r):
    """The motion of a camera in short statements (computed for runs prepared before they were saved)"""
    return r.get('motion_items') or (motion_items(r) if 'flags' in r else [])


def popup(r):
    """The popup of a camera: the verdict and its summary, short statements of what the review saw and of the
    motion of the video, the links to the video, the details (folded) and the tile"""
    e = html.escape
    fd, rec = r['findings'], r['record'] or {}
    ul = lambda items: '<ul>' + ''.join(f'<li>{e(i)}</li>' for i in items) + '</ul>' if items else ''  # noqa: E731
    parts = [f"<b>{e(r['camera_id'])}</b> {e(r.get('name') or '')}"]
    details = []
    if fd:
        parts.append(f"<br><i>{e(NAMES.get(fd.get('severity')))}</i>, {e(str(fd.get('confidence')))} confidence"
                     f"<br>{e(fd.get('summary', ''))}")
        shaking, seen = fd.get('shaking') or {}, []
        if shaking.get('visible') in ('yes', 'uncertain'):
            seen.append(f"Shaking: {shaking['visible']}")
            details.append(f"<b>Shaking</b>: {e('; '.join(shaking.get('effects') or []))}")
        if shaking.get('intensity_mmi'):
            bound = shaking.get('mmi_kind') == 'upper bound'
            seen.append(f"Intensity: MMI {'≤ ' if bound else ''}{shaking['intensity_mmi']}"
                        + (f" ({shaking['mmi_confidence']} % confidence)" if shaking.get('mmi_confidence') is not None
                           else ''))
            details.append(f"<b>Intensity</b>: {e(shaking.get('basis', ''))}")
        for key in DETECTIONS:
            item = fd.get(key) or {}
            if item.get('present') in ('yes', 'uncertain'):
                label = TAG_LABELS.get(key, key.replace('_', ' '))
                seen.append(f"{label}: {item['present']}")
                details.append(f"<b>{e(label)}</b>: {e(item.get('description', ''))} {e(item.get('where', ''))}")
        parts.append('<div class="h">Review</div>' + (ul(seen) or '<ul><li>nothing detected</li></ul>'))
        details.append(f"<b>Reviewed by</b> {e(str(rec.get('reviewed_by')))} {e(str(rec.get('model') or ''))}")
    elif rec.get('error'):
        parts.append(f"<br><i>review failed: {e(rec['error'])}</i>")
    else:
        parts.append('<br><i>not reviewed</i>')
    if items := items_of(r):
        parts.append('<div class="h">Video motion</div>' + ul(items))
    if r.get('motion_text'):
        details.append(f"<b>Video motion</b>: {e(r['motion_text'])}")
    if links := video_links(r):
        parts.append(f'<div>{links}</div>')
    if (r.get('vibration') or {}).get('plot'):
        arr = (f" &middot; <a href=\"{e(r['arrivals']['plot'])}\" target=\"_blank\">arrivals</a> "
               f"({e(r['arrivals']['quality'])}, <a href=\"arrivals.html\">map</a>)" if r.get('arrivals') else '')
        parts.append(f"<div>Vibration: <a href=\"{e(r['vibration']['plot'])}\" target=\"_blank\">figure</a> &middot; "
                     f'<a href="vibration.html">map</a>{arr}</div>')
    if details:
        parts.append('<details><summary>Details</summary>' + '<br>'.join(details) + '</details>')
    if r.get('tile'):
        parts.append(f"<a href=\"{e(r['tile'])}\" target=\"_blank\"><img src=\"{e(r['tile'])}\" width=\"380\"></a>")
    return '<div class="pop">' + ''.join(parts) + '</div>'


def title_of(event):
    return ' - '.join(str(event[k]) for k in ('event_id', 'place') if event.get(k)) or 'camera survey'


def event_popup(event):
    """The popup of the epicenter: the event, its magnitude, place, origin time and depth, a note, and a link to
    its page (event.json: url, e.g., the USGS event page, where the magnitude may since have been revised)"""
    e = html.escape
    mag = event.get('magnitude')
    head = ' '.join(x for x in (f"M{mag:g}" if isinstance(mag, (int, float)) else '', event.get('magnitude_type') or '')
                    if x)
    lines = [f"<b>{e(', '.join(x for x in (head, event.get('place') or '') if x))}</b>"]
    when = (event.get('origin_utc') or '').replace('T', ' ').rstrip('Z')[:19]
    if when:
        depth = event.get('depth_km')
        lines.append(f'{e(when)} UTC' + (f', depth {depth:g} km' if depth is not None else ''))
    if event.get('note'):
        lines.append(e(event['note']))
    if event.get('event_id'):
        lines.append(f"<span style=\"color:#52514e\">{e(event['event_id'])}</span>")
    if event.get('url'):
        lines.append(f'<a href="{e(event["url"])}" target="_blank" rel="noopener">Event page (latest magnitude and '
                     f'location)</a>')
    return '<div class="pop">' + '<br>'.join(lines) + '</div>'


def map_html(rows, event, out):
    points = [dict(lat=r['lat'], lon=r['lon'], color=COLORS.get(r['severity'], COLORS[None]), label=r['camera_id'],
                   popup=popup(r), tags=tags(r)) for r in rows if r.get('lat') is not None]
    tag_list = [dict(key=k, label=TAG_LABELS.get(k, k.replace('_', ' ').capitalize()),
                     yes=sum(p['tags'].get(k) == 'yes' for p in points),
                     uncertain=sum(p['tags'].get(k) == 'uncertain' for p in points)) for k in TAGS]
    epi = [event['lat'], event['lon']] if event.get('lat') is not None else None
    title = html.escape(title_of(event))
    legend = ''.join(f'<div><span style="background:{COLORS[k]}"></span>{html.escape(v)}</div>'
                     for k, v in NAMES.items())
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .legend{{background:#fff;padding:6px 8px;font:12px sans-serif}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px}}
.filter{{background:#fff;padding:6px 10px;font:12px sans-serif;line-height:1.6;max-height:70vh;overflow:auto}}
.filter label{{display:block;cursor:pointer;white-space:nowrap}} .filter .zero{{color:#999}}
.filter .n{{color:#555;margin-left:4px}} .filter hr{{margin:4px 0;border:0;border-top:1px solid #ddd}}
.pop{{font:12px/1.4 sans-serif}} .pop .h{{margin-top:6px;font-weight:bold;color:#444}}
.pop ul{{margin:2px 0 4px;padding-left:18px}} .pop li{{margin:0}}
.pop details{{margin:4px 0;color:#333}} .pop summary{{cursor:pointer;color:#1a5fb4}} .pop img{{margin-top:4px}}
.leaflet-popup-content img{{max-width:100%;height:auto}}
.legend .ph{{margin-bottom:2px}} {PANEL_CSS}
@media (max-width:768px){{.pop,.filter,.legend{{font-size:14px}} .filter{{max-height:45vh}}}}</style>
</head><body><div id="map"></div><script>
const points = {json.dumps(points)};
const tags = {json.dumps(tag_list)};
const epi = {json.dumps(epi)};
const map = L.map('map');
{BASEMAPS_JS}{PANEL_JS}const bounds = [];
for (const p of points) {{
  p.marker = L.circleMarker([p.lat, p.lon], {{radius: 8, color: '#222', weight: 1, fillColor: p.color, fillOpacity: 0.9}})
    .bindPopup(p.popup, popupOptions(420)).bindTooltip(p.label).addTo(map);
  bounds.push([p.lat, p.lon]);
}}
// the filter: the cameras with any of the checked tags are shown (a dashed outline if only uncertain);
// the others are dimmed, or hidden
const filter = L.control({{position: 'topleft'}});
filter.onAdd = () => {{
  const d = L.DomUtil.create('div', 'filter');
  const esc = s => String(s).replace(/[&<>"]/g, c => ({{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}})[c]);
  d.innerHTML = '<div class="ph">Show cameras with</div><div class="pb">' + tags.map(t =>
      `<label class="${{t.yes + t.uncertain ? '' : 'zero'}}"><input type="checkbox" data-tag="${{t.key}}"> ` +
      `${{esc(t.label)}}<span class="n">${{t.yes}}${{t.uncertain ? ' + ' + t.uncertain + '?' : ''}}</span></label>`).join('') +
    '<hr><label><input type="checkbox" id="f-uncertain" checked> include uncertain (?)</label>' +
    '<label><input type="checkbox" id="f-hide"> hide the other cameras</label></div>';
  d.addEventListener('change', update);
  return foldable(d, !small);
}};
filter.addTo(map);
function update() {{
  const checked = [...document.querySelectorAll('.filter input[data-tag]:checked')].map(i => i.dataset.tag);
  const uncertain = document.getElementById('f-uncertain').checked;
  const hide = document.getElementById('f-hide').checked;
  for (const p of points) {{
    const states = checked.map(k => p.tags[k]).filter(s => s === 'yes' || (uncertain && s === 'uncertain'));
    const match = !checked.length || states.length > 0;
    const sure = !checked.length || states.includes('yes');
    if (match) {{
      p.marker.setStyle({{opacity: 1, fillOpacity: 0.9, weight: sure ? 1 : 2, dashArray: sure ? null : '3 2'}});
      p.marker.setRadius(8);
    }} else {{
      p.marker.setStyle({{opacity: 0.35, fillOpacity: 0.15, weight: 1, dashArray: null}});
      p.marker.setRadius(5);
    }}
    if (match || !hide) {{ if (!map.hasLayer(p.marker)) p.marker.addTo(map); }} else p.marker.remove();
    if (match && checked.length && map.hasLayer(p.marker)) p.marker.bringToFront();
  }}
}}
if (epi) {{ L.marker(epi).bindPopup({json.dumps(event_popup(event))}).addTo(map); bounds.push(epi); }}
{FIT_JS}const legend = L.control({{position: 'bottomright'}});
legend.onAdd = () => {{ const d = L.DomUtil.create('div', 'legend');
  d.innerHTML = '<div class="ph">Legend</div><div class="pb">' + {json.dumps(legend)} + '</div>';
  return foldable(d, !small); }};
legend.addTo(map);
</script></body></html>
"""
    with open(os.path.join(out, 'map.html'), 'w') as f:
        f.write(page)


def map_png(rows, event, out):
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    pts = [r for r in rows if r.get('lat') is not None]
    lats = [r['lat'] for r in pts] + ([event['lat']] if event.get('lat') is not None else [])
    lons = [r['lon'] for r in pts] + ([event['lon']] if event.get('lon') is not None else [])
    mid = (min(lons) + max(lons)) / 2 if lons else 0
    for r in sorted(pts, key=lambda r: -1 if r['severity'] is None else r['severity']):
        ax.scatter(r['lon'], r['lat'], s=70, c=COLORS.get(r['severity'], COLORS[None]), edgecolors='#222',
                   linewidths=0.6, zorder=3)
        if (r['severity'] or 0) >= 2:                   # labels toward the middle, to stay inside the map
            east = r['lon'] > mid
            ax.annotate(f"{r['camera_id']}: {', '.join(detected(r['findings']))[:70]}", (r['lon'], r['lat']),
                        xytext=(-7 if east else 7, 5), textcoords='offset points', ha='right' if east else 'left',
                        fontsize=7, zorder=4, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    if event.get('lat') is not None:
        ax.scatter(event['lon'], event['lat'], marker='*', s=260, c='k', zorder=6)
        ax.annotate(f"M{event.get('magnitude', '')}", (event['lon'], event['lat']), xytext=(8, -12),
                    textcoords='offset points', fontsize=9, zorder=6)
    if lats:
        ax.set_aspect(1 / math.cos(math.radians(float(np.mean(lats)))), adjustable='datalim')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    for k, v in NAMES.items():
        ax.scatter([], [], c=COLORS[k], edgecolors='#222', linewidths=0.6, label=v)
    ax.legend(fontsize=7, loc='best', frameon=True)
    ax.set_xlabel('longitude', fontsize=9)
    ax.set_ylabel('latitude', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(title_of(event), fontsize=10, loc='left')
    fig.savefig(os.path.join(out, 'map.png'), dpi=130)
    plt.close(fig)


def report(out, stages=('final', 'review', 'screen'), mask=True, vibration=False):
    """The table and the maps; first, unless mask is False, the people blurred in the tiles and frames from the
    boxes of the review (eqcam.mask); with vibration, the vibration of the cameras and of the objects in their
    views, for scientific use (eqcam.vibration: vibration.html, and the figures linked from the map)"""
    with open(os.path.join(out, 'event.json')) as f:
        event = json.load(f)
    blurred = mask_run(out, stages) if mask else {}
    rows = merge(out, stages)
    for r in rows:
        r['people_blurred'] = blurred.get(r['camera_id'])
    if vibration:
        from .vibration import vibration as analyze_vibration
        analyze_vibration(out, rows, event)
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
