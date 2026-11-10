"""Download the OpenForensics dataset (Le et al., ICCV 2021, CC BY 4.0) from Zenodo.

https://zenodo.org/records/5528418

The full dataset is about 67 GB of zips, so choose the parts you need in
FILES_TO_GET (or pass --all). Downloads resume if interrupted, are checked
against Zenodo's MD5, and are then extracted one zip at a time.

Usage:
    python download_openforensics.py
    python download_openforensics.py --data-dir D:/OpenForensics --keep-zips
    python download_openforensics.py --all
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

RECORD_ID = "5528418"

# Starter subset, about 3 GB (Val only).
# Train_part_1..5.zip: training images (~20 GB total)
# Val.zip: validation images (~3 GB)
# Test-Dev_part_1..2.zip: test images (~8 GB)
# Test-Challenge_part_1..5.zip: harder test set with perturbations (~23 GB)
# *_poly.json: COCO-style annotations (real/fake label, box and mask per face)
FILES_TO_GET = [
    "readme.txt",
    "Val.zip",
    "Val_poly.json",
    # Add these later if you need more training data:
    # "Train_part_1.zip",
    # "Train_poly.json",
]


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(info, dest):
    size = info["size"]
    url = info["links"]["self"]
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        return
    if have > size:
        dest.unlink()
        have = 0

    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r:
        if have and r.status != 206:  # server ignored the Range header, start over
            have = 0
        with open(dest, "ab" if have else "wb") as f, tqdm(
            total=size, initial=have, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/OpenForensics"),
                        help="where extracted images/annotations go")
    parser.add_argument("--zip-dir", type=Path, default=Path("zips"),
                        help="where zips are downloaded")
    parser.add_argument("--keep-zips", action="store_true",
                        help="keep zips after extraction (default: delete them to save disk)")
    parser.add_argument("--all", action="store_true", help="download every file in the record")
    args = parser.parse_args()

    data_dir, zip_dir = args.data_dir, args.zip_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)
    print(f"Free space on disk: {shutil.disk_usage(data_dir).free / 1e9:.0f} GB")

    # List the files in the Zenodo record
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD_ID}") as r:
        record = json.load(r)
    files = {f["key"]: f for f in record["files"]}
    for name in sorted(files):
        print(f"{name:30s} {files[name]['size'] / 1e9:6.2f} GB")
    print(f"Total: {sum(f['size'] for f in files.values()) / 1e9:.1f} GB")

    to_get = sorted(files) if args.all else FILES_TO_GET

    total = sum(files[n]["size"] for n in to_get)
    largest_zip = max((files[n]["size"] for n in to_get if n.endswith(".zip")), default=0)
    # Zips are extracted and deleted one at a time, so peak use is all extracted data plus one zip
    needed = total + (0 if args.keep_zips else largest_zip)
    free = shutil.disk_usage(data_dir).free
    print(f"Selected {len(to_get)} files, {total / 1e9:.1f} GB")
    print(f"Need about {needed / 1e9:.1f} GB on disk, {free / 1e9:.0f} GB free")
    if needed > free:
        print("WARNING: not enough space, remove some files from FILES_TO_GET")

    # Download, verify and extract
    for name in to_get:
        info = files[name]
        dest = (zip_dir if name.endswith(".zip") else data_dir) / name
        extracted_marker = data_dir / f".{name}.done"
        if extracted_marker.exists():
            print(f"{name}: already downloaded and extracted")
            continue

        for _ in range(3):
            download(info, dest)
            if md5sum(dest) == info["checksum"].split(":", 1)[1]:
                print(f"{name}: OK")
                break
            print(f"{name}: checksum mismatch, re-downloading")
            dest.unlink()
        else:
            raise RuntimeError(f"{name} failed checksum 3 times")

        if name.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                for member in tqdm(z.infolist(), desc=f"Extracting {name}"):
                    z.extract(member, data_dir)
            if not args.keep_zips:
                dest.unlink()
        extracted_marker.touch()

    print("Contents of", data_dir.resolve())
    for p in sorted(data_dir.iterdir()):
        if not p.name.startswith((".", "_")):
            print(" ", p.name)

    # Quick look at the annotations
    val_path = data_dir / "Val_poly.json"
    if val_path.exists():
        with open(val_path) as f:
            val = json.load(f)
        print("Keys:", list(val.keys()))
        print("Images:", len(val["images"]), "| Faces:", len(val["annotations"]))
        print("Categories:", val["categories"])


if __name__ == "__main__":
    main()
