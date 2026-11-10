"""eqcam: rapid survey of camera videos after an earthquake with vision-language models

  eqcam prepare   the motion of each video, and a tile of its frames before, during and after the shaking
  eqcam review    the findings of each camera from a backend (claude, gemini or qwen)
  eqcam validate  check findings written by hand or in a Claude Code session
  eqcam report    the table and the maps
  eqcam run       prepare, review (and escalate), report

examples:
  eqcam prepare --cameras videos.csv --videos /data/event123 --event event.json --out runs/event123
  eqcam review --out runs/event123 --backend claude --model claude-haiku-5-5 --stage screen
  eqcam review --out runs/event123 --backend claude --stage final --escalate-from screen
  eqcam report --out runs/event123
  eqcam run --cameras videos.csv --videos /data/event123 --event event.json --out runs/event123 --backend qwen
"""

import argparse
import sys

from .backends import BACKENDS, make_backend
from .frames import prepare
from .report import report
from .review import review, validate_stage


def add_prepare_args(p):
    p.add_argument('--cameras', required=True, help='the cameras and their videos (.csv: camera_id, name, lat, lon, '
                   'video, and optionally start_utc, view)')
    p.add_argument('--videos', default='.', help='the folder the video paths are relative to (default: %(default)s)')
    p.add_argument('--event', help='the event (.json: event_id, origin_utc, lat, lon, depth_km, magnitude, place)')
    p.add_argument('--only', nargs='+', help='only these camera ids')
    p.add_argument('--motion-fps', type=float, default=15, help='frame rate for the motion (default: %(default)s)')
    p.add_argument('--motion-width', type=int, default=320, help='width for the motion, px (default: %(default)s)')
    p.add_argument('--max-seconds', type=float, default=600, help='longest part of a video to decode, s '
                   '(default: %(default)s)')


def add_backend_args(p):
    p.add_argument('--backend', choices=sorted(BACKENDS), required=True)
    p.add_argument('--model', help='the model (defaults: claude-opus-5-5, gemini-2.5-flash, '
                   'Qwen/Qwen3-VL-8B-Instruct)')
    p.add_argument('--platform', choices=['anthropic', 'vertex'], help='claude: through the Anthropic API '
                   '(default) or Google Vertex AI')
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'], help='claude: the effort '
                   '(default medium)')
    p.add_argument('--project', help='Vertex AI: the Google Cloud project (default: $GOOGLE_CLOUD_PROJECT)')
    p.add_argument('--region', help='claude on Vertex AI: the region (default global)')
    p.add_argument('--location', help='gemini: the Vertex AI location (default global)')
    p.add_argument('--base-url', help='qwen: the OpenAI-compatible server (default: $EQCAM_QWEN_URL or '
                   'http://localhost:8000/v1)')
    p.add_argument('--workers', type=int, default=8, help='requests in parallel (default: %(default)s)')
    p.add_argument('--stage', default='review', help='the name of this review stage (default: %(default)s)')
    p.add_argument('--escalate-from', help='review only the cameras this earlier stage flagged')
    p.add_argument('--min-severity', type=int, default=2, help='with --escalate-from: the severity that flags '
                   'a camera (low confidence and errors always do; default: %(default)s)')
    p.add_argument('--overwrite', action='store_true', help='review again the cameras that have findings')


def backend_from(args, model=None):
    return make_backend(args.backend, model=model or args.model, platform=args.platform, effort=args.effort,
                        project=args.project, region=args.region, location=args.location, base_url=args.base_url)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='eqcam', description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='the motion and the frames for review')
    add_prepare_args(p)
    p.add_argument('--out', required=True, help='the output folder')
    r = sub.add_parser('review', help='the findings from a vision-language backend')
    r.add_argument('--out', required=True, help='the folder written by prepare')
    r.add_argument('--only', nargs='+', help='only these camera ids')
    add_backend_args(r)
    v = sub.add_parser('validate', help='check the findings of a stage')
    v.add_argument('--out', required=True)
    v.add_argument('--stage', default='review')
    o = sub.add_parser('report', help='the table and the maps')
    o.add_argument('--out', required=True)
    o.add_argument('--stages', nargs='+', default=['final', 'review', 'screen'],
                   help='the stages to take the findings from, in order of preference (default: %(default)s)')
    a = sub.add_parser('run', help='prepare, review, report')
    add_prepare_args(a)
    a.add_argument('--out', required=True)
    add_backend_args(a)
    a.add_argument('--escalate-model', help='a second model for the cameras the first one flags (stage final)')
    args = parser.parse_args(argv)

    if args.command == 'prepare':
        prepare(args.cameras, args.videos, args.out, args.event, args.only, motion_fps=args.motion_fps,
                motion_width=args.motion_width, max_seconds=args.max_seconds)
    elif args.command == 'review':
        review(args.out, backend_from(args), args.stage, args.workers, args.only, args.escalate_from,
               args.min_severity, args.overwrite)
    elif args.command == 'validate':
        problems = validate_stage(args.out, args.stage)
        for name, issues in problems.items():
            print(f'{name}: ' + '; '.join(issues))
        print(f'{len(problems)} files with problems' if problems else 'all findings valid')
        sys.exit(1 if problems else 0)
    elif args.command == 'report':
        report(args.out, args.stages)
    else:
        prepare(args.cameras, args.videos, args.out, args.event, args.only, motion_fps=args.motion_fps,
                motion_width=args.motion_width, max_seconds=args.max_seconds)
        review(args.out, backend_from(args), args.stage, args.workers, args.only, None, args.min_severity,
               args.overwrite)
        stages = [args.stage]
        if args.escalate_model:
            review(args.out, backend_from(args, args.escalate_model), 'final', args.workers, args.only, args.stage,
                   args.min_severity, args.overwrite)
            stages.insert(0, 'final')
        report(args.out, stages)
    return 0


if __name__ == '__main__':
    sys.exit(main())
