"""Summarise results/*.csv into the Markdown tables used in README.md."""
import csv
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
NAMES = {"naive": "Naive lawn-mower (no safety layer)",
         "reactive": "Reactive detect-and-avoid (same CBF, binary map)",
         "belief": "**Probabilistic barrier (proposed)**"}


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    s = load("survey_metrics.csv")
    seeds = sorted({r["seed"] for r in s})
    lines = [f"**Survey**, {len(seeds)} seeds (mean; triggers are totals):", "",
             "| Policy | Virtual triggers | Coverage after survey | POD | Survey time (s) | Distance (m) |",
             "|---|---|---|---|---|---|"]
    for pol in ("naive", "reactive", "belief"):
        rows = [r for r in s if r["policy"] == pol]
        mean = lambda k: sum(float(r[k]) for r in rows) / len(rows)  # noqa: E731
        trig = sum(int(r["virtual_triggers"]) for r in rows)
        haz = sum(int(r["hazards"]) for r in rows)
        lines.append(f"| {NAMES[pol]} | {trig} / {haz} devices | {mean('coverage'):.0%} | "
                     f"{mean('pod'):.2f} | {mean('time_s'):.0f} | {mean('distance_m'):.0f} |")
    c = load("chain_metrics.csv")
    tot = lambda k: sum(int(r[k]) for r in c)  # noqa: E731
    mean_c = lambda k: sum(float(r[k]) for r in c) / len(c)  # noqa: E731
    import json
    with open(os.path.join(OUT, "chain_log.json"), encoding="utf-8") as f:
        logs = json.load(f)
    found_late = sum(1 for v in logs.values() for r in v["records"]
                     if r.get("pass") == 2 and not r.get("second_look")
                     and r.get("truth_kind") not in ("clutter", "none")
                     and r.get("result") == "processed")
    lines += ["", "**Last-metre chain + mop-up** (proposed policy, all seeds):", "",
              "| Devices processed | of which found only in the mop-up | Placed within spec | Contacts | Verification correct / wrong / inconclusive | Clutter & false alarms resolved | Devices missed at identification | of which re-flagged on second look | Coverage after mop-up | Triggers (whole mission) |",
              "|---|---|---|---|---|---|---|---|---|---|",
              f"| {tot('processed')} | {found_late} | {tot('placed_ok')} | {tot('contacts')} | "
              f"{tot('verified_correct')} / {tot('verified_wrong')} / {tot('inconclusive')} | "
              f"{tot('false_alarms_resolved')} | {tot('missed_at_identification')} | "
              f"{tot('recovered_on_second_look')} | {mean_c('final_coverage'):.1%} | "
              f"{sum(int(r['virtual_triggers_total']) for r in c)} |"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
