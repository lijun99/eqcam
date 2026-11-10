"""A demo event, with no API: synthetic videos at made-up camera locations in Los Angeles, made-up findings of
every severity, and the table and maps, to see what eqcam produces

usage: python examples/demo.py [--out runs/demo]
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from test_eqcam import write_video                     # noqa: E402 - a textured scene, shaken, then changed
from eqcam import backends, frames, report, review, schema   # noqa: E402

# id, name, lat, lon, shaken, changed after, severity, detections, summary (all made up)
CAMERAS = [
    ('i10_downtown', 'I-10 at downtown interchange', 34.040, -118.270, True, True, 3,
     {'structural_damage': 'Parapet and facade of a building collapsed onto the sidewalk.',
      'smoke_or_dust': 'Dust cloud rising from the collapse.'},
     'Facade collapse with a dust cloud next to the freeway.'),
    ('hollywood_hills', 'Ridge camera above Hollywood', 34.125, -118.330, True, True, 3,
     {'landslide': 'New scar and debris on the slope below the road.'},
     'Fresh landslide on the slope below the road.'),
    ('port_la', 'Port of Los Angeles terminal', 33.740, -118.270, True, True, 2,
     {'ground_failure': 'Possible sand boils and settlement near the wharf edge.',
      'water': 'Water sloshing in the slip.'},
     'Possible liquefaction near the wharf; water sloshing.'),
    ('pasadena_710', 'SR-710 at Pasadena', 34.140, -118.150, True, True, 2,
     {'smoke_or_dust': 'Gray plume behind the buildings, not in BEFORE.', 'fire': 'Glow at the base of the plume.'},
     'Possible fire: new smoke plume with a glow.'),
    ('santa_monica_pier', 'Santa Monica pier stream', 34.008, -118.498, True, False, 1,
     {'people_response': 'About 30 people moving off the pier.'},
     'Strong swaying of lamp posts; people leaving the pier.'),
    ('burbank_5', 'I-5 at Burbank', 34.180, -118.310, True, False, 1, {}, 'Poles and signs swayed; no damage seen.'),
    ('long_beach_405', 'I-405 at Long Beach', 33.820, -118.180, True, False, 0, {}, 'Light shaking, nothing notable.'),
    ('glendale_2', 'SR-2 at Glendale (no video)', 34.160, -118.240, False, False, None, {}, ''),
]
TABLE = {c[0]: c for c in CAMERAS}


class MadeUp:
    """A stand-in backend that returns the made-up findings of each camera"""
    name, model = 'demo', 'made-up findings'

    def review(self, image, text):
        cid = text.split('Camera: ')[1].split(' - ')[0]
        _, _, _, _, shaken, _, severity, found, summary = TABLE[cid]
        fd = {key: {'present': 'no', 'description': '', 'where': ''} for key in schema.DETECTIONS}
        for key, description in found.items():
            fd[key] = {'present': 'yes' if severity == 3 else 'uncertain', 'description': description,
                       'where': 'AFTER frame'}
        fd.update(camera_status='ok', severity=severity, confidence='high' if severity == 3 else 'medium',
                  summary=summary,
                  shaking={'visible': 'yes' if shaken else 'no',
                           'effects': ['lamp posts swaying'] if severity else [],
                           'intensity_mmi': {3: 'VIII', 2: 'VII', 1: 'V-VI', 0: 'III-IV'}[severity],
                           'basis': 'made up for the demo'})
        return backends.Result(fd, self.model, {'input_tokens': 0, 'output_tokens': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out', default='runs/demo', help='the folder of the demo (default: %(default)s)')
    args = parser.parse_args()
    videos = os.path.join(args.out, 'videos')
    os.makedirs(videos, exist_ok=True)
    with open(os.path.join(args.out, 'videos.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['camera_id', 'name', 'lat', 'lon', 'video', 'start_utc', 'view'])
        for k, (cid, name, lat, lon, shaken, changed, *_) in enumerate(CAMERAS):
            if cid != 'glendale_2':
                path = os.path.join(videos, f'{cid}.mp4')
                if not os.path.exists(path):
                    write_video(path, seconds=12, shake=(4.0, 8.0) if shaken else (100, 100),
                                change_at=8.0 if changed else 100, seed=k)
            w.writerow([cid, name, lat, lon, f'{cid}.mp4', '', ''])
    with open(os.path.join(args.out, 'event.json'), 'w') as f:
        json.dump({'event_id': 'demo', 'place': 'M6.8 made-up demo event, Los Angeles',
                   'origin_utc': '2026-01-01T12:00:00Z', 'lat': 34.05, 'lon': -118.25, 'depth_km': 10,
                   'magnitude': 6.8}, f)
    run = os.path.join(args.out, 'run')
    frames.prepare(os.path.join(args.out, 'videos.csv'), videos, run, os.path.join(args.out, 'event.json'))
    review.review(run, MadeUp(), workers=4, overwrite=True)
    report.report(run)
    print(f'open {os.path.join(run, "map.html")} in a browser')


if __name__ == '__main__':
    main()
