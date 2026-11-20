"""mask: blur the people in the frames of a run, from the boxes the review returned (people_boxes)

The review sees the tile (up to four frames); its boxes, 0-1000 of the tile, are padded and blurred in the tile
and, mapped through the tile layout saved by prepare, in the full frames. The frames as prepared are kept in
cameras/<id>/original/, and each masking starts again from them: a new review masks again, and a camera whose
boxes are gone gets its frames back. Pillow only, a few milliseconds per camera.
"""

import json
import os
import shutil

from PIL import Image, ImageFilter

from .review import load_record

PAD = 0.1          # margin around each box, as a fraction of its size
MIN_PAD_PX = 4     # and at least this many pixels of the tile


def tile_boxes(boxes, size):
    """The boxes (0-1000 of the tile) in pixels of the tile, padded and clipped; malformed ones are skipped"""
    w, h = size
    out = []
    for b in boxes or []:
        try:
            x0, y0, x1, y1 = (float(b[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max'))
        except (KeyError, TypeError, ValueError):
            continue
        x0, x1 = sorted((x0 * w / 1000, x1 * w / 1000))
        y0, y1 = sorted((y0 * h / 1000, y1 * h / 1000))
        px, py = max(MIN_PAD_PX, PAD * (x1 - x0)), max(MIN_PAD_PX, PAD * (y1 - y0))
        box = (max(0.0, x0 - px), max(0.0, y0 - py), min(float(w), x1 + px), min(float(h), y1 + py))
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


def frame_boxes(boxes, place, frame_size):
    """The parts of the tile boxes over a frame placed at place = [x0, y0, width, height] in the tile, in pixels
    of the full frame"""
    px, py, pw, ph = place
    fw, fh = frame_size
    out = []
    for x0, y0, x1, y1 in boxes:
        ix0, iy0, ix1, iy1 = max(x0, px), max(y0, py), min(x1, px + pw), min(y1, py + ph)
        if ix1 > ix0 and iy1 > iy0:
            out.append(((ix0 - px) * fw / pw, (iy0 - py) * fh / ph, (ix1 - px) * fw / pw, (iy1 - py) * fh / ph))
    return out


def blur(im, boxes):
    """Pixelate and blur the boxes of an image, in place"""
    for box in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        w, h = x1 - x0, y1 - y0
        if w < 2 or h < 2:
            continue
        region = im.crop((x0, y0, x1, y1))
        region = region.resize((max(1, w // 8), max(1, h // 8)), Image.BILINEAR).resize((w, h), Image.NEAREST)
        im.paste(region.filter(ImageFilter.GaussianBlur(max(2.0, min(w, h) / 6))), (x0, y0))
    return im


def mask_camera(out, m, boxes):
    """Blur the people of one camera in its tile and frames; the number of boxes blurred, or None if the frames
    could not be mapped (a run prepared before the tile layout was saved: only the tile is masked)"""
    if not m.get('tile'):
        return 0
    cdir = os.path.join(out, 'cameras', m['camera_id'])
    orig = os.path.join(cdir, 'original')
    files = [m['tile']] + [f['file'] for f in (m.get('frames') or {}).values()]
    if not os.path.isdir(orig):
        if not boxes:
            return 0
        os.makedirs(orig)
        for f in files:
            shutil.copy2(os.path.join(out, f), os.path.join(orig, os.path.basename(f)))
    with Image.open(os.path.join(orig, os.path.basename(m['tile']))) as im:
        tile_im = im.convert('RGB')
    tb = tile_boxes(boxes, tile_im.size)
    blur(tile_im, tb).save(os.path.join(out, m['tile']), quality=90)
    layout = m.get('tile_layout')
    for key, f in (m.get('frames') or {}).items():
        with Image.open(os.path.join(orig, os.path.basename(f['file']))) as im:
            frame = im.convert('RGB')
        fb = frame_boxes(tb, layout[key], frame.size) if layout and key in layout else []
        blur(frame, fb).save(os.path.join(out, f['file']), quality=92)
    return len(tb) if layout or not tb else None


def mask_run(out, stages=('final', 'review', 'screen')):
    """Blur the people of every camera from the boxes of the first stage that reviewed it; {camera_id: boxes}"""
    with open(os.path.join(out, 'metrics.json')) as f:
        metrics = json.load(f)
    counts = {}
    for m in metrics:
        record = next((r for s in stages if (r := load_record(out, s, m['camera_id'])) is not None), None)
        boxes = ((record or {}).get('findings') or {}).get('people_boxes') or []
        counts[m['camera_id']] = mask_camera(out, m, boxes)
    unmapped = [c for c, n in counts.items() if n is None]
    n = sum(v or 0 for v in counts.values())
    print(f'people blurred: {n} boxes in {sum(1 for v in counts.values() if v)} cameras'
          + (f'; frames not masked (prepare again): {", ".join(unmapped)}' if unmapped else ''))
    return counts
