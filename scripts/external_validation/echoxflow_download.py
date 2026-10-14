"""Download and extract the exams in selected_exams.csv (resumable: skips exams already extracted)."""
import csv
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(r"C:\Claude\Doppler\external\echoxflow")
URL = "https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/main/exams/{}.tar"
rows = list(csv.DictReader(open(ROOT / "selected_exams.csv")))
(ROOT / "exams").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)
for i, r in enumerate(rows):
    ex = r["exam_id"]
    if (ROOT / "data" / "exams" / ex).exists():
        continue
    tar = ROOT / "exams" / f"{ex}.tar"
    subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(tar), URL.format(ex)], check=True)
    with tarfile.open(tar) as tf:
        tf.extractall(ROOT / "data", filter="data")
    tar.unlink()  # keep only the extracted Zarr stores
    print(f"{i + 1}/{len(rows)} {ex} {r['size_gb']} GB", flush=True)
print("done")
