"""Choose EchoXFlow exams to download (<= budget GB): most annotated PW/CW spectral recordings per GB.

Writes selected_exams.csv (exam_id, size_gb, n_pw, n_cw, n_pw_ann, n_cw_ann).
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Claude\Doppler\external\echoxflow")
BUDGET_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 19.5

d = json.load(open(ROOT / "croissant.json", encoding="utf8"))
rs = {r.get("@id", r.get("name")): r for r in d["recordSet"]}
rec = [{k.split("/")[-1]: v for k, v in r.items()} for r in rs["recordings"]["data"]]
sizes = {e["path"].split("/")[-1].removesuffix(".tar"): e["size"] for e in json.load(open(ROOT / "exams_tree.json"))}

stats = collections.defaultdict(lambda: collections.Counter())
for r in rec:
    ex = r["exam_id"]
    paths = r.get("array_paths") or []
    for kind, key in (("pw", "1d_pulsed_wave_doppler"), ("cw", "1d_continuous_wave_doppler")):
        if f"data/{key}" in paths:
            stats[ex][f"n_{kind}"] += 1
            if any(p.startswith(f"data/{key}_annotation_") for p in paths):
                stats[ex][f"n_{kind}_ann"] += 1
                stats[ex][f"n_{kind}_ann_items"] += sum(p.startswith(f"data/{key}_annotation_") for p in paths)

rows = []
for ex, c in stats.items():
    if ex not in sizes:
        continue
    gb = sizes[ex] / 1e9
    ann = c["n_pw_ann"] + c["n_cw_ann"]
    if ann == 0:
        continue
    rows.append(dict(exam_id=ex, size_gb=round(gb, 3), n_pw=c["n_pw"], n_cw=c["n_cw"], n_pw_ann=c["n_pw_ann"],
                     n_cw_ann=c["n_cw_ann"], score=ann / gb))
rows.sort(key=lambda r: -r["score"])
print(f"{len(rows)} exams with annotated spectral Doppler; total {sum(r['size_gb'] for r in rows):.0f} GB")
sel, tot = [], 0.0
for r in rows:
    if tot + r["size_gb"] <= BUDGET_GB:
        sel.append(r)
        tot += r["size_gb"]
print(f"selected {len(sel)} exams, {tot:.2f} GB, PW recordings with annotations {sum(r['n_pw_ann'] for r in sel)},"
      f" CW {sum(r['n_cw_ann'] for r in sel)}")
with open(ROOT / "selected_exams.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0]))
    w.writeheader()
    w.writerows(sel)
for r in sel[:8]:
    print(r)
