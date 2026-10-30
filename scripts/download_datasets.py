"""Download the public datasets for Paper 1 into $WHSSL_DATA/raw (default data/raw).

Waves (every source verified on 2026-10-02; sizes are compressed download sizes):
  wave 1  first wave, about 3.4 GB: FOG contest (128 PD patients, with test labels), GSTRIDE, Huntington's,
          PADS, HAR70+, Daphnet, Oxford ssl-wearables weights
  wave 2  large open sets, about 151 GB: FOG daily living (62 GB, unlabelled, in-domain pretraining),
          LTMM (22 GB), Mobilise-D TVS (58 GB), Mobilise-D CVS outcomes (1 GB), Voisard clinical gait (2.2 GB),
          Capture-24 (6.9 GB)
Not automated (need a personal login): WearGait-PD on Synapse; see docs/paper1_ssl/Datasets.md.

Masoud approved downloading all planned datasets on the new machine (2026-10-02).

Behaviour
  * Resumable: partial files are kept as .part and resumed where the server supports byte ranges.
  * Parallel: files of 64 MB or more are fetched as byte-range pieces over --connections connections (default 8).
    Zenodo serves about 0.5 MB/s per connection (measured 2026-10-03: 1 connection 0.5 MB/s, 4 gave 2.1 MB/s),
    so one connection would need about 70 h for wave 2. Piece progress is kept in <file>.part.json.
  * Verified: Zenodo MD5s are checked; S3 objects are checked by size.
  * Idempotent: a file already recorded in data/raw/MANIFEST.json with the same size is skipped
    (fixes the 2026-10-01 bug where sources without a published size were re-downloaded every run).
  * PhysioNet sets (PADS, LTMM) come from the physionet-open S3 mirror; physionet.org zips crawl at 0.1 MB/s.
  * Uses Python's TLS stack (on the old laptop, Windows curl failed certificate revocation checks).
  * Zips are extracted next to themselves (folder named after the zip) after download.

Usage
  python scripts/download_datasets.py --list                 # plan only: sources, files, sizes
  python scripts/download_datasets.py --wave 1               # first wave
  python scripts/download_datasets.py --wave 1 2             # everything automated
  python scripts/download_datasets.py --only ltmm pads       # selected sources
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("WHSSL_DATA", ROOT / "data")) / "raw"
UA = {"User-Agent": "whssl-research (academic"
                     + (f"; {os.environ['WHSSL_CONTACT_EMAIL']}" if os.environ.get("WHSSL_CONTACT_EMAIL") else "") + ")"}
S3 = "https://physionet-open.s3.amazonaws.com"
PARALLEL_MIN = 64 << 20  # files at least this large are fetched as parallel byte-range pieces
PIECE_MIN, PIECE_MAX = 8 << 20, 128 << 20  # bytes one connection fetches per request


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def zenodo(record: str, keep=None):
    d = json.loads(_get(f"https://zenodo.org/api/records/{record}"))
    out = []
    for f in d["files"]:
        if keep and f["key"] not in keep:
            continue
        ck = f.get("checksum", "")
        out.append({"url": f["links"]["self"], "name": f["key"], "size": f["size"],
                    "md5": ck.split(":", 1)[1] if ck.startswith("md5:") else None})
    return out


def figshare(article: str):
    d = json.loads(_get(f"https://api.figshare.com/v2/articles/{article}"))
    return [{"url": f["download_url"], "name": f["name"], "size": f["size"], "md5": f.get("computed_md5") or None}
            for f in d["files"]]


def s3_prefix(prefix: str, max_files=None):
    """Every object under a physionet-open prefix (paginated listing). Names keep the sub-path."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        x = _get(f"{S3}/?{urllib.parse.urlencode(q)}").decode()
        for key, size in zip(re.findall(r"<Key>(.*?)</Key>", x), re.findall(r"<Size>(\d+)</Size>", x)):
            if key.endswith("/"):
                continue
            out.append({"url": f"{S3}/{urllib.parse.quote(key)}", "name": key[len(prefix):], "size": int(size), "md5": None})
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m or (max_files and len(out) >= max_files):
            break
        token = m.group(1)
    return out[:max_files] if max_files else out


def single(url, name, size=None, md5=None):
    return [{"url": url, "name": name, "size": size, "md5": md5}]


SOURCES = {  # name: (wave, lister, note)
    "fog_contest": (1, lambda: zenodo("12565292", keep={"Competition Dataset.zip"}),
                    "Zenodo 10.5281/zenodo.12565292, CC BY 4.0 (bundled LICENSE says Apache-2.0)"),
    "gstride": (1, lambda: zenodo("17052815"), "Zenodo 17052815, CC BY 4.0"),
    "huntington": (1, lambda: zenodo("14384260"), "Zenodo 14384260, CC BY 4.0"),
    "pads": (1, lambda: s3_prefix("parkinsons-disease-smartwatch/1.0.0/"), "PhysioNet PADS 1.0.0 via S3, CC BY-NC-SA 4.0"),
    "har70plus": (1, lambda: single("https://archive.ics.uci.edu/static/public/780/har70.zip", "har70.zip"), "UCI 780, CC BY 4.0"),
    "daphnet": (1, lambda: single("https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip", "daphnet.zip"),
                "UCI 245, CC BY 4.0"),
    "oxford_ssl_wearables": (1, lambda: single("https://codeload.github.com/OxWearables/ssl-wearables/zip/refs/heads/main",
                                               "ssl-wearables-main.zip"),
                             "harnet weights; ACADEMIC USE ONLY, cite 'Copyright (c) 2022, University of Oxford'"),
    "fog_daily_living": (2, lambda: zenodo("12565292", keep={"Daily Living Data Part 1.zip", "Daily Living Data Part 2.zip"}),
                         "65 PD patients x 7 days, unlabelled; 45 are also DeFOG patients (exclude per fold)"),
    "ltmm": (2, lambda: s3_prefix("ltmm/1.0.0/"), "PhysioNet LTMM 1.0.0 via S3, ODC-By 1.0"),
    "mobilised_tvs": (2, lambda: zenodo("15861907"), "Zenodo 15861907 (newest TVS record), CC BY 4.0"),
    "mobilised_cvs": (2, lambda: zenodo("21060991"), "Zenodo 21060991, derived outcomes only, CC BY 4.0 per Zenodo"),
    "voisard": (2, lambda: figshare("28806086"), "figshare 28806086, CC BY per the API (survey said NC-ND: check page)"),
    "capture24": (2, lambda: single("https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/download_file"
                                    "?file_format=&safe_filename=capture24.zip&type_of_work=Dataset", "capture24.zip", None),
                  "Oxford ORA, CC BY 4.0, about 6.9 GB; server ignores byte ranges (restarts from zero)"),
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ranges_supported(url: str) -> bool:
    """True if the server answers a byte-range request with 206. Network errors propagate (a rerun resumes)."""
    with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"}), timeout=60) as r:
        return r.status == 206


def fetch_single(item: dict, part: Path, log, quiet: bool, label: str) -> None:
    """One connection; resumes a partial .part where the server honours byte ranges."""
    start = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if start:
        headers["Range"] = f"bytes={start}-"
    t0, last = time.time(), time.time()
    with urllib.request.urlopen(urllib.request.Request(item["url"], headers=headers), timeout=120) as r:
        if start and r.status != 206:  # server ignored the range: start over
            start = 0
        done = start
        with open(part, "ab" if start else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if not quiet and time.time() - last > 30:
                    pct = f"{100 * done / item['size']:.0f}% " if item["size"] else ""
                    log(f"  {label}: {done / 1e9:.2f} GB {pct}({(done - start) / 1e6 / (time.time() - t0):.1f} MB/s)")
                    last = time.time()


def fetch_parallel(item: dict, part: Path, connections: int, log, quiet: bool, label: str) -> None:
    """Byte-range pieces over several connections, written in place into a .part preallocated to full size.

    Progress per piece ([start, end, bytes done]) is saved to <file>.part.json every few seconds and only counts
    bytes already flushed, so a killed run resumes safely. A .part left by a single-connection attempt (no
    .json) is kept as an already-downloaded prefix. The caller still checks the MD5.
    """
    size = item["size"]
    side = part.with_name(part.name + ".json")
    pieces = None
    if side.exists() and part.exists() and part.stat().st_size == size:
        state = json.loads(side.read_text())
        if state.get("size") == size:
            pieces = state["pieces"]
    if pieces is None:
        prefix = min(part.stat().st_size, size) if part.exists() else 0
        step = max(PIECE_MIN, min(PIECE_MAX, -(-(size - prefix) // (4 * connections))))
        pieces = ([[0, prefix, prefix]] if prefix else []) + [[s, min(s + step, size), 0] for s in range(prefix, size, step)]
        with open(part, "ab") as f:
            f.truncate(size)

    def save():
        tmp = side.with_name(side.name + ".tmp")
        tmp.write_text(json.dumps({"url": item["url"], "size": size, "pieces": pieces}))
        tmp.replace(side)

    todo, errors = queue.Queue(), []
    for k, (s, e, d) in enumerate(pieces):
        if s + d < e:
            todo.put(k)

    def worker():
        # A piece that keeps failing is given up alone; the other connections carry on, and a rerun resumes it.
        # (Until 2026-10-03 the first give-up stopped every connection: when another download saturated the
        # line, Zenodo pieces timed out, and the whole 31 GB file wound down to a stop at 21%.)
        with open(part, "r+b") as f:
            while True:
                try:
                    k = todo.get_nowait()
                except queue.Empty:
                    return
                s, e = pieces[k][0], pieces[k][1]
                fails = 0
                while s + pieces[k][2] < e:
                    pos = before = s + pieces[k][2]
                    try:
                        req = urllib.request.Request(item["url"], headers={**UA, "Range": f"bytes={pos}-{e - 1}"})
                        with urllib.request.urlopen(req, timeout=120) as r:
                            if r.status != 206:
                                raise IOError(f"server ignored the byte range (HTTP {r.status})")
                            f.seek(pos)
                            while pos < e:
                                chunk = r.read(min(1 << 20, e - pos))
                                if not chunk:
                                    break
                                f.write(chunk)
                                f.flush()
                                pos += len(chunk)
                                pieces[k][2] = pos - s
                        err = "connection closed early"
                    except Exception as ex:  # timeout, dropped connection, 429 or 5xx: back off, resume the piece
                        err = str(ex)
                    if s + pieces[k][2] >= e:
                        break
                    fails = 0 if s + pieces[k][2] > before else fails + 1  # any progress resets the count
                    if fails >= 8:
                        errors.append(f"bytes {s}-{e}: {err}")
                        break
                    time.sleep(min(120, 5 * 2 ** fails))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(connections, todo.qsize()))]
    done0, t0, last = sum(p[2] for p in pieces), time.time(), time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        save()
        if not quiet and time.time() - last > 30:
            done = sum(p[2] for p in pieces)
            log(f"  {label}: {done / 1e9:.2f} GB {100 * done / size:.0f}% "
                f"({(done - done0) / 1e6 / (time.time() - t0):.1f} MB/s over {len(threads)} connections)")
            last = time.time()
    save()
    missing = sum(e - s - d for s, e, d in pieces)
    if missing:
        raise IOError(f"{label}: {missing / 1e6:.0f} MB still missing ({errors[0] if errors else 'no error'}); rerun to resume")


def fetch(item: dict, dest: Path, known: dict, log, quiet=False, connections=1) -> dict:
    target = dest / item["name"]
    rel = str(target.relative_to(RAW)).replace("\\", "/")
    rec = known.get(rel) or {}
    if not target.exists() and rec.get("archive_removed") and target.suffix == ".zip" and target.with_suffix("").is_dir():
        # the archive was deleted on purpose after its extracted folder was verified member by member (MANIFEST)
        return {**rec, "status": "extracted; archive removed"}
    if target.exists():
        size = target.stat().st_size
        rec = known.get(rel)
        if (item["size"] and size == item["size"]) or (not item["size"] and rec and rec.get("size") == size):
            if not item["md5"] or (rec and rec.get("md5") == item["md5"]) or md5sum(target) == item["md5"]:
                return {**item, "path": rel, "size": size, "md5": item["md5"] or (rec or {}).get("md5"), "status": "present"}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    side = part.with_name(part.name + ".json")
    if connections > 1 and (item["size"] or 0) >= PARALLEL_MIN and ranges_supported(item["url"]):
        fetch_parallel(item, part, connections, log, quiet, target.name)
    else:
        if side.exists():  # a parallel attempt preallocated this .part, so its length is not progress
            part.unlink(missing_ok=True)
            side.unlink()
        fetch_single(item, part, log, quiet, target.name)
    got = part.stat().st_size
    if item["size"] and got != item["size"]:
        raise IOError(f"size mismatch for {rel}: {got} != {item['size']}")
    digest = md5sum(part) if (item["md5"] or got < 5e9) else None
    if item["md5"] and digest != item["md5"]:
        part.unlink()
        side.unlink(missing_ok=True)
        raise IOError(f"MD5 mismatch for {rel} (partial file deleted; rerun to retry)")
    part.replace(target)
    side.unlink(missing_ok=True)
    if not quiet:
        log(f"  done: {rel} {got / 1e9:.3f} GB")
    return {**item, "path": rel, "size": got, "md5": digest, "status": "downloaded",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, nargs="*", default=[1])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--threads", type=int, default=8, help="parallel downloads for many-file sources")
    ap.add_argument("--connections", type=int, default=8,
                    help="byte-range connections per file of 64 MB or more (1 = the old single-connection behaviour)")
    ap.add_argument("--no-unzip", action="store_true")
    args = ap.parse_args()
    names = [n for n, (w, _, _) in SOURCES.items() if (args.only and n in args.only) or (not args.only and w in args.wave)]

    def log(m):
        print(m, flush=True)

    if args.list:
        total = 0
        for n in names:
            w, lister, note = SOURCES[n]
            items = lister()
            size = sum(i["size"] or 0 for i in items)
            total += size
            unknown = sum(1 for i in items if not i["size"])
            log(f"wave {w}  {n:22s} {len(items):6d} files {size / 1e9:8.2f} GB{' + unknown' if unknown else ''}   {note}")
        log(f"total (known sizes): {total / 1e9:.1f} GB -> {RAW}")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    mpath = RAW / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    failures = []
    for n in names:
        w, lister, note = SOURCES[n]
        log(f"[{n}] wave {w}: {note}")
        known = {e["path"]: e for e in manifest.get(n, []) if "path" in e}
        try:
            items = lister()
            dest = RAW / n
            if len(items) > 20:  # many small files (S3 mirrors): parallel, compact log
                with cf.ThreadPoolExecutor(args.threads) as ex:
                    futs = [ex.submit(fetch, i, dest, known, log, True) for i in items]
                    entries, errs = [], []
                    for k, fu in enumerate(cf.as_completed(futs), 1):
                        try:
                            entries.append(fu.result())
                        except Exception as e:  # keep going, report at the end
                            errs.append(str(e))
                        if k % 500 == 0:
                            log(f"  {k}/{len(items)} files")
                if errs:
                    raise IOError(f"{len(errs)} of {len(items)} files failed, e.g. {errs[0]}")
                entries.sort(key=lambda e: e["path"])
                log(f"  {len(entries)} files, {sum(e['size'] for e in entries) / 1e9:.2f} GB present")
            else:  # few large files: one at a time, each over parallel byte ranges
                entries = [fetch(i, dest, known, log, connections=args.connections) for i in items]
            manifest[n] = entries
            mpath.write_text(json.dumps(manifest, indent=1))
        except Exception as e:  # keep going; a rerun resumes
            log(f"  FAILED: {e}")
            failures.append(n)

    if not args.no_unzip:
        for n in names:
            for e in manifest.get(n, []):
                p = RAW / e["path"]
                if p.suffix == ".zip" and p.exists() and not p.with_suffix("").exists():
                    log(f"unzip {e['path']}")
                    with zipfile.ZipFile(p) as z:
                        z.extractall(p.with_suffix(""))
    log("FAILED: " + ", ".join(failures) if failures else "ALL DONE")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
