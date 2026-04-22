"""
Batch-run bird identification on all images in a folder (no ground-truth labels).
Outputs CSV to stdout and optional --out file.

Usage:
  python scripts/batch_scan.py path/to/Testing_Photo
  python scripts/batch_scan.py path/to/Testing_Photo --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_settings
from core.classifier_service import ClassifierService
from core.deepseek_client import DeepSeekClient
from core.image_pipeline import prepare_image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch bird ID scan (no labels).")
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=ROOT / "Testing_Photo",
        help="Folder containing images (default: ./Testing_Photo)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write CSV to this path.")
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    settings = load_settings()
    client = DeepSeekClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
        retry_count=settings.retry_count,
        openrouter_provider_ignore=settings.openrouter_provider_ignore,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_title=settings.openrouter_title,
    )
    classifier = ClassifierService(client=client, top_k=settings.top_k)

    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS
    )
    if not files:
        print(f"No images found in {folder} (supported: {', '.join(sorted(EXTS))})")
        sys.exit(0)

    rows: list[dict[str, str]] = []
    for path in files:
        row: dict[str, str] = {
            "file": path.name,
            "top1": "",
            "top1_conf": "",
            "top2": "",
            "top2_conf": "",
            "top3": "",
            "top3_conf": "",
            "error": "",
        }
        try:
            prepared = prepare_image(
                path.read_bytes(),
                max_upload_mb=settings.max_upload_mb,
                max_image_side=settings.max_image_side,
            )
            result = classifier.classify(prepared["data_url"])
            preds = result.get("predictions") or []
            for i, pred in enumerate(preds[:3], start=1):
                label = pred.get("common_name") or pred.get("species") or ""
                conf = float(pred.get("confidence", 0.0))
                row[f"top{i}"] = label
                row[f"top{i}_conf"] = f"{conf:.4f}"
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)[:500]
        rows.append(row)

    fieldnames = [
        "file",
        "top1",
        "top1_conf",
        "top2",
        "top2_conf",
        "top3",
        "top3_conf",
        "error",
    ]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}")

    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
