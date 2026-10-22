"""Check that every held-out-domain run really tested on its held-out domain.

    python scripts/verify_xdomain.py --results "D:/projects/Fall before impact/results/xdomain"

For each run: the saved config must say mode "holdout", and every clip in
outcomes_test.csv must belong to the protocol's test domain. Exits non-zero if any
run fails, so nothing is exported or published from a run that tested elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

DOMAINS = {
    "urfd_to_le2i": lambda cid: cid.startswith("le2i_"),
    "le2i_to_urfd": lambda cid: cid.startswith("urfd_"),
    "leave_coffee_room_out": lambda cid: cid.startswith("le2i_coffee_room"),
    "leave_home_out": lambda cid: cid.startswith("le2i_home"),
}


def protocol_of(config: str) -> str:
    name = config[3:] if config.startswith("xd_") else config
    return name.rsplit("_", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)

    bad, checked = [], 0
    with (root / "runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        run_dir, protocol = Path(row["run_dir"]), protocol_of(row["config"])
        problems = []
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        mode = cfg["data"]["split"].get("mode")
        if mode != "holdout":
            problems.append(f"saved config has split mode {mode!r}, not 'holdout'")
        inside = DOMAINS[protocol]
        with (run_dir / "outcomes_test.csv").open(newline="", encoding="utf-8") as handle:
            ids = [r.get("clip_id") or r.get("clip") for r in csv.DictReader(handle)]
        outside = [i for i in ids if i and not inside(i)]
        if outside:
            problems.append(f"{len(outside)} of {len(ids)} test clips are outside the domain, e.g. {outside[:2]}")
        if not ids:
            problems.append("no test clips recorded")
        checked += 1
        if problems:
            bad.append((row["config"], row["seed"], problems))

    for config, seed, problems in bad:
        print(f"FAIL {config} seed {seed}: " + "; ".join(problems))
    print(f"{checked - len(bad)} of {checked} runs tested on their held-out domain")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
