import argparse
import json
from masbench.longhorizon.closed_loop import cases_from_jsonl, run_closed_loop_suite

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="MASbench/data")
parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
parser.add_argument("--out", default="MASbench/results/longhorizon/closed_loop_test_1_0")
parser.add_argument("--deadline-scale", type=float, default=1.0)
args = parser.parse_args()
report = run_closed_loop_suite(cases_from_jsonl(args.data_dir, args.split), args.out, args.deadline_scale)
print(json.dumps(report, indent=2, sort_keys=True))
