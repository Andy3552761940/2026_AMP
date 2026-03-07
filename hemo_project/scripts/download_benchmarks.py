#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import requests

DATASETS = {
    "hemopi_home": "https://webs.iiitd.edu.in/raghava/hemopi/",
    "hemopi_dataset_page": "https://webs.iiitd.edu.in/raghava/hemopi/dataset.php",
    "hemopi2_home": "https://webs.iiitd.edu.in/raghava/hemopi2/",
    "dbaasp": "https://dbaasp.org/",
}


def check_url(url: str, timeout: int = 15) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        return {"url": url, "status_code": r.status_code, "ok": r.ok}
    except Exception as e:
        return {"url": url, "status_code": None, "ok": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = {k: check_url(v) for k, v in DATASETS.items()}
    guide_path = out_dir / "download_guide.json"
    guide_path.write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "# Benchmark 下载指引",
        "",
        "自动检查结果见 download_guide.json。",
        "如果状态码为 200 但仍无法下载，通常是站点需要手工点击下载按钮。",
        "",
    ]
    for name, url in DATASETS.items():
        txt.append(f"- {name}: {url}")

    (out_dir / "README_download.txt").write_text("\n".join(txt), encoding="utf-8")
    print(f"Saved download metadata to: {guide_path}")


if __name__ == "__main__":
    main()
