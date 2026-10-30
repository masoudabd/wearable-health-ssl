"""Download the parts of the FORGE release that Masoud approved on 2026-10-04 (paper/DECISIONS.md, decision 21).

Sources (Hugging Face, public, no sign-in):
  model    huggingface.co/Liornis/forge-fog            encoders/ (three pretrained encoders, MIT licence),
                                                       classification/ (the 33 released downstream DeFOG detectors,
                                                       added 2026-10-28 for scripts/forge_detector.py) and the
                                                       card files (README.md, manifest.yaml, checksums.json)
  dataset  huggingface.co/datasets/Liornis/fog-dataset fogathome_dailyliving/ (labeled naturalistic daily living,
                                                       11 patients) and fogathome/ (structured home protocol, 12
                                                       patients), and README.md; non-commercial research use under
                                                       the source studies' data-use terms, no re-identification
Not downloaded: the copies of the contest data (71 GB) already on disk.
Approved size: about 8.4 GB (paper/notes/forge_check_2026-10-04.md) plus the 2.36 GB of classification heads.

Every file is checked against the size and, for files stored with Git LFS, the SHA-256 that the Hugging Face API
lists; a rerun resumes partial files and skips verified ones. Files go to data/raw/forge/{model,dataset}/<path>, and
data/raw/MANIFEST.json gets a "forge" entry per file.

Usage: .venv/bin/python scripts/download_forge.py [--list] [--workers 8] [--part model dataset]

--part limits the run to one repository; the MANIFEST entry then keeps the other repository's recorded files
(a full run rewrites the whole "forge" entry).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("WHSSL_DATA", ROOT / "data")) / "raw"
DEST = RAW / "forge"
REPOS = {  # name: (api path, resolve path, prefixes to take)
    "model": ("models/Liornis/forge-fog", "Liornis/forge-fog",
              ("encoders/", "classification/", "README.md", "manifest.yaml", "checksums.json")),
    "dataset": ("datasets/Liornis/fog-dataset", "datasets/Liornis/fog-dataset",
                ("fogathome_dailyliving/", "fogathome/", "README.md")),
}
LICENCE = {"model": "MIT (FORGE code and weights)",
           "dataset": "non-commercial research use under the source studies' data-use terms; no re-identification"}


def list_files(api_path: str, prefixes: tuple[str, ...]) -> list[dict]:
    url = f"https://huggingface.co/api/{api_path}/tree/main?recursive=true"
    out = []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out += [x for x in r.json() if x.get("type") == "file"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""), flags=re.I)
        url = m.group(1) if m else None
    keep = [x for x in out if any(x["path"] == p or (p.endswith("/") and x["path"].startswith(p)) for p in prefixes)]
    return [{"path": x["path"], "size": (x.get("lfs") or {}).get("size", x.get("size")),
             "sha256": (x.get("lfs") or {}).get("oid")} for x in keep]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, item: dict) -> dict:
    target = DEST / repo / item["path"]
    url = f"https://huggingface.co/{REPOS[repo][1]}/resolve/main/{item['path']}"
    rec = {"url": url, "name": Path(item["path"]).name, "size": item["size"], "sha256": item["sha256"],
           "path": str(target.relative_to(RAW)), "licence": LICENCE[repo]}
    if target.exists() and target.stat().st_size == item["size"] and (not item["sha256"]
                                                                      or sha256sum(target) == item["sha256"]):
        return {**rec, "status": "present"}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code != 206:  # no resume support: start again
                    have, headers = 0, {}
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                with open(part, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
    got = part.stat().st_size
    if got != item["size"]:
        raise IOError(f"size mismatch for {item['path']}: {got} != {item['size']}")
    if item["sha256"] and sha256sum(part) != item["sha256"]:
        part.unlink()
        raise IOError(f"SHA-256 mismatch for {item['path']} (partial file deleted; rerun to retry)")
    part.replace(target)
    return {**rec, "status": "downloaded", "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list files and sizes, download nothing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part", nargs="+", choices=sorted(REPOS), default=sorted(REPOS),
                    help="repositories to fetch (default: both)")
    a = ap.parse_args()
    items = [(repo, it) for repo, (api, _, prefixes) in REPOS.items() if repo in a.part
             for it in list_files(api, prefixes)]
    total = sum(it["size"] or 0 for _, it in items)
    print(f"{len(items)} files, {total / 1e9:.2f} GB", flush=True)
    if a.list:
        for repo in a.part:
            sub = [it for r, it in items if r == repo]
            print(f"  {repo}: {len(sub)} files, {sum(it['size'] or 0 for it in sub) / 1e9:.3f} GB")
        return
    if total > 11.5e9:
        raise SystemExit(f"{total / 1e9:.2f} GB is more than the approved about 10.7 GB "
                         "(8.4 GB plus the 2.36 GB of classification heads): check the listing first")
    t0, done, entries, failed = time.time(), 0, [], []
    with ThreadPoolExecutor(a.workers) as ex:
        futures = {ex.submit(fetch, repo, it): (repo, it) for repo, it in items}
        for i, fut in enumerate(as_completed(futures), 1):
            repo, it = futures[fut]
            try:
                e = fut.result()
                entries.append(e)
                done += it["size"] or 0
            except Exception as exc:  # report and continue; a rerun retries the failed files
                failed.append(f"{repo}/{it['path']}: {exc}")
            if i % 200 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} files, {done / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    mpath = RAW / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    kept = [e for e in manifest.get("forge", [])
            if Path(e["path"]).parts[1] not in a.part] if set(a.part) != set(REPOS) else []
    manifest["forge"] = sorted(kept + entries, key=lambda e: e["path"])
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(entries)} entries to {mpath}; {len(failed)} failed")
    for f in failed[:20]:
        print("  FAILED", f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
