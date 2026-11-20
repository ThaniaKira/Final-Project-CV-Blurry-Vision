"""Prepare a genuine HaGRID Kaggle subset into subject-disjoint six-class splits.

Kaggle dataset: innominate817/hagrid-sample-30k-384p
Expected layout (inside archive or extracted directory):
  hagrid-sample-30k-384p/ann_train_val/{palm,fist,peace,like,one,ok}.json
  hagrid-sample-30k-384p/hagrid_30k/train_val_{class}/{image_id}.jpg

Unlike the 20 original cartoon drawings, these are real hand photographs.
The code NEVER evaluates/trains on source-identical people across splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

KAGGLE_SLUG = "innominate817/hagrid-sample-30k-384p"
SOURCE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"
CLASS_MAP = {
    "palm": "Open_Hand", "fist": "Fist", "peace": "Peace",
    "like": "Thumbs_Up", "one": "Pointing", "ok": "OK",
}
SPLITS = ("train", "val", "test")


def find_archive(raw: Path) -> Path | None:
    return next(iter(sorted(raw.glob("*.zip"))), None)


def download_using_kaggle(raw: Path) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_SLUG}. Approx. 1 GB; free disk space required.", flush=True)
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
                        "-p", str(raw), "--force"], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Kaggle CLI is missing. Install it with: python -m pip install kaggle\n"
                           "Alternatively, download the dataset ZIP in a browser and run "
                           "--archive path/to/file.zip") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Kaggle download failed. Check internet/Kaggle authentication or "
                           "download the ZIP manually from " + SOURCE_URL) from exc
    result = find_archive(raw)
    if result is None:
        raise RuntimeError(f"No dataset ZIP in {raw}")
    return result


class ArchiveReader:
    """Read either the original Kaggle ZIP or its extracted directory."""
    def __init__(self, archive: Path | None, source_dir: Path | None):
        self.zipfile = zipfile.ZipFile(archive) if archive else None
        self.root = source_dir
        names = self.zipfile.namelist() if self.zipfile else [
            str(p.relative_to(self.root)).replace("\\", "/") for p in self.root.rglob("*") if p.is_file()
        ]
        self.by_suffix = {name: name for name in names}
        self.names = names

    def match(self, suffix: str):
        return next((s for s in self.names if s.endswith(suffix)), None)

    def read(self, name: str):
        return self.zipfile.read(name) if self.zipfile else (self.root / name).read_bytes()

    def close(self):
        if self.zipfile:
            self.zipfile.close()


def stable_split_people(records, seed: int = 42):
    """70/15/15-ish subject-disjoint assignment, deterministic and shared by classes.

    Group identity is HaGRID's ``user_id``. All photos of one person must
    belong to the same partition. Class balance is CHECKED after splitting.
    """
    groups = sorted(set(row["user_id"] for row in records))
    if len(groups) < 20:
        raise ValueError(f"Only {len(groups)} independent subjects; obtain more real photos")
    random.Random(seed).shuffle(groups)
    n_train = int(round(len(groups) * .70))
    n_val = int(round(len(groups) * .15))
    assignment = {}
    for i, group in enumerate(groups):
        assignment[group] = ("train" if i < n_train else "val" if i < n_train + n_val else "test")
    return assignment


def choose_bbox(annotation, target):
    candidates = []
    for bbox, label in zip(annotation.get("bboxes", []), annotation.get("labels", [])):
        if str(label).strip().lower() != target:
            continue
        if len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        if min(w, h) <= 0 or any(not np.isfinite(v) for v in (x, y, w, h)):
            continue
        candidates.append((w * h, (x, y, w, h)))
    return max(candidates)[1] if candidates else None


def crop_hand(raw_bytes, normalized_bbox, pad_ratio=.30):
    image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    x, y, w, h = normalized_bbox
    # HaGRID boxes are normalized top-left x, top-left y, width, height.
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = max(w * width, h * height) * (1 + 2 * pad_ratio)
    x1 = int(max(0, round(cx - side / 2)))
    x2 = int(min(width, round(cx + side / 2)))
    y1 = int(max(0, round(cy - side / 2)))
    y2 = int(min(height, round(cy + side / 2)))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    crop = image[y1:y2, x1:x2]
    max_size = 360
    scale = min(1., max_size / max(crop.shape[:2]))
    return cv2.resize(crop, None, fx=scale, fy=scale) if scale < 1 else crop


def prepare(archive=None, source_dir=None, output="data/hagrid_real", max_per_class=400, seed=42):
    if bool(archive) == bool(source_dir):
        raise ValueError("Choose exactly one of --archive OR --source-dir")
    source = ArchiveReader(Path(archive) if archive else None, Path(source_dir) if source_dir else None)
    rng = random.Random(seed)
    records, missing = [], []
    try:
        for hagrid_class, label in CLASS_MAP.items():
            ann_name = source.match(f"/ann_train_val/{hagrid_class}.json")
            if ann_name is None:
                ann_name = source.match(f"ann_train_val/{hagrid_class}.json")
            if ann_name is None:
                missing.append(f"ann_train_val/{hagrid_class}.json")
                continue
            annotation_dict = json.loads(source.read(ann_name))
            image_ends = f"/train_val_{hagrid_class}/"
            image_lookup = {Path(name).stem: name for name in source.names if image_ends in name and name.lower().endswith((".jpg", ".jpeg", ".png"))}
            candidates = []
            for photo_id, annotation in annotation_dict.items():
                if photo_id not in image_lookup:
                    continue
                bbox = choose_bbox(annotation, hagrid_class)
                if bbox is None:
                    continue
                person = annotation.get("user_id")
                if not person:
                    # Without subjects, no honest source-disjoint evaluation.
                    raise RuntimeError(f"Missing user_id annotation for {photo_id}; cannot safely split by subject")
                candidates.append({"id": photo_id, "user_id": str(person), "class": label,
                                   "hagrid_class": hagrid_class, "bbox": bbox, "source": image_lookup[photo_id]})
            rng.shuffle(candidates)
            selected = candidates if max_per_class == 0 else candidates[:max_per_class]
            records += selected
            print(f"{hagrid_class:6s} -> {label:10s}: {len(selected)} / {len(candidates)} usable annotated images")
        if missing:
            raise RuntimeError("HaGRID archive structure mismatch; missing " + ", ".join(missing))
        if any(sum(r["class"] == c for r in records) < 20 for c in CLASS_MAP.values()):
            raise RuntimeError("Not enough images per class to train/evaluate; use a higher --max-per-class")
        assignment = stable_split_people(records, seed)
        output = Path(output)
        # Guard against silently mixing old and new data.
        if output.exists() and any(output.rglob("*.jpg")):
            raise RuntimeError(f"{output} already has images. Use a fresh directory to avoid mixing dataset versions")
        output.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        manifest = []
        for row in records:
            split = assignment[row["user_id"]]
            image = crop_hand(source.read(row["source"]), row["bbox"])
            if image is None:
                continue
            dest = output / split / row["class"] / f"{row['hagrid_class']}_{row['id']}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed writing {dest}")
            counts[(split, row["class"])] += 1
            manifest.append({"split": split, "class": row["class"], "hagrid_class":row["hagrid_class"],
                             "image_id": row["id"], "user_id":row["user_id"],
                             "filename": str(dest).replace("\\", "/")})
        for split in SPLITS:
            for label in CLASS_MAP.values():
                if counts[(split, label)] < 5:
                    raise RuntimeError(f"Not enough {label} in {split}: {counts[(split,label)]}; increase --max-per-class")
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
        info = {"source":SOURCE_URL,"dataset":KAGGLE_SLUG,"source_license":"CC BY-SA 4.0 (verify the Kaggle dataset terms)",
                "prepared_n":len(manifest),"sample_limit_per_source_class":max_per_class,
                "split_method":"Source subject (HaGRID user_id); globally disjoint across all six classes",
                "split_seed":seed,"counts":{split:{label:counts[(split,label)] for label in CLASS_MAP.values()} for split in SPLITS},
                "bbox_crop":"Square bbox with 30% padding, capped at 360px; MediaPipe real-hand detection",
                "warning":"The original 20 illustrated samples are NOT mixed into this dataset."}
        (output / "data_card.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print("Prepared:", output.resolve(), "\n", json.dumps(info["counts"],indent=2))
        return info
    finally:
        source.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive", help="Path to downloaded Kaggle ZIP")
    g.add_argument("--source-dir", help="Extracted HaGRID dataset directory")
    g.add_argument("--download", action="store_true", help="Download Kaggle HaGRID ZIP through Kaggle CLI")
    p.add_argument("--max-per-class", type=int, default=400, help="0=all matched images; default 400/class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/hagrid_real")
    args = p.parse_args()
    archive = args.archive or (str(download_using_kaggle(Path("data/kaggle_raw"))) if args.download else None)
    prepare(archive=archive,source_dir=args.source_dir,output=args.output,max_per_class=args.max_per_class,seed=args.seed)
