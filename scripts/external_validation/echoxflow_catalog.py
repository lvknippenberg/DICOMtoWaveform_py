"""Explore the EchoXFlow croissant catalogue: which recordings hold spectral (PW/CW) Doppler, what arrays they have,
and which exams have the most of them (to choose a <=20 GB download)."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Claude\Doppler\external\echoxflow")
d = json.load(open(ROOT / "croissant.json", encoding="utf8"))
print(d["description"][:3000])
rs = {r["@id"] if "@id" in r else r["name"]: r for r in d["recordSet"]}
rec = rs["recordings"]["data"]
arr = rs["arrays"]["data"]


def strip(row):
    return {k.split("/")[-1]: v for k, v in row.items()}


rec = [strip(r) for r in rec]
arr = [strip(a) for a in arr]
print("\nexample recording:", json.dumps(rec[0])[:800])
modes = collections.Counter()
ctypes = collections.Counter()
for r in rec:
    m = r.get("modes")
    for x in (m if isinstance(m, list) else [m]):
        modes[str(x)] += 1
    c = r.get("content_types")
    for x in (c if isinstance(c, list) else [c]):
        ctypes[str(x)] += 1
print("\nmodes:", modes.most_common(40))
print("\ncontent types:", ctypes.most_common(60))
paths = collections.Counter(a.get("array_path") for a in arr)
print("\narray paths (top 80):")
for p, n in paths.most_common(80):
    print(f"  {n:6d} {p}")
roles = collections.Counter((a.get("role"), str(a.get("units")), str(a.get("coordinate_system"))) for a in arr)
print("\nroles/units:", roles.most_common(60))
if len(sys.argv) > 1:
    key = sys.argv[1]
    for a in arr:
        if key in str(a.get("array_path")):
            print(json.dumps(a)[:600])
            break
