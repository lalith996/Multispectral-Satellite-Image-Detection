"""
download_eurosat_ms.py
Downloads a balanced subset of REAL Copernicus Sentinel-2 13-band GeoTIFF patches
directly from the official EuroSAT-MS Zenodo archive (DOI: 10.5281/zenodo.7711810)
using HTTP Range requests — fetching only the requested 1,000 GeoTIFFs (~85 MB)
instead of the entire 2.1 GB zip archive.
"""

from __future__ import annotations
import os
import struct
import sys
import urllib.request
import zlib
from collections import defaultdict

ZENODO_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "SpectralCG-Forensics/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def get_content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SpectralCG-Forensics/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers["Content-Length"])


def parse_zip_central_directory(url: str) -> list[dict]:
    """Read only the End-of-Central-Directory and Central Directory from remote ZIP via Range requests."""
    total_size = get_content_length(url)
    print(f"Remote EuroSAT_MS.zip size: {total_size / (1024**3):.2f} GB")

    # Fetch last 65 KB to locate EOCD
    tail_size = min(65536, total_size)
    tail = http_range(url, total_size - tail_size, total_size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos == -1:
        raise RuntimeError("Could not locate ZIP End of Central Directory (EOCD).")

    eocd = tail[eocd_pos:eocd_pos + 22]
    _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack("<4sHHHHIIH", eocd)

    # Check for ZIP64 locator if offset/size is 0xFFFFFFFF
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        loc_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
        if loc_pos == -1:
            raise RuntimeError("ZIP64 indicated but ZIP64 locator not found.")
        _, _, z64_offset, _ = struct.unpack("<4sIQI", tail[loc_pos:loc_pos + 20])
        z64_eocd = http_range(url, z64_offset, z64_offset + 55)
        cd_size, cd_offset = struct.unpack("<QQ", z64_eocd[40:56])

    print(f"Fetching ZIP Central Directory ({cd_size / (1024**2):.2f} MB)...")
    cd_bytes = http_range(url, cd_offset, cd_offset + cd_size - 1)

    entries = []
    pos = 0
    while pos < len(cd_bytes):
        if cd_bytes[pos:pos + 4] != b"PK\x01\x02":
            break
        (
            _, _, _, _, comp_method, _, _, _, comp_size, uncomp_size,
            fname_len, extra_len, comment_len, _, _, _, local_hdr_offset
        ) = struct.unpack("<4sHHHHHHIIIHHHHHII", cd_bytes[pos:pos + 46])
        fname = cd_bytes[pos + 46:pos + 46 + fname_len].decode("utf-8", errors="replace")
        extra = cd_bytes[pos + 46 + fname_len:pos + 46 + fname_len + extra_len]

        # Handle ZIP64 extra field if needed
        if local_hdr_offset == 0xFFFFFFFF or comp_size == 0xFFFFFFFF:
            ep = 0
            while ep + 4 <= len(extra):
                hdr_id, sz = struct.unpack("<HH", extra[ep:ep + 4])
                if hdr_id == 0x0001:
                    vals = struct.unpack(f"<{sz // 8}Q", extra[ep + 4:ep + 4 + sz])
                    vi = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = vals[vi]; vi += 1
                    if comp_size == 0xFFFFFFFF:
                        comp_size = vals[vi]; vi += 1
                    if local_hdr_offset == 0xFFFFFFFF:
                        local_hdr_offset = vals[vi]; vi += 1
                    break
                ep += 4 + sz

        entries.append({
            "name": fname,
            "comp_method": comp_method,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "offset": local_hdr_offset,
        })
        pos += 46 + fname_len + extra_len + comment_len

    return entries


def extract_entry(url: str, entry: dict) -> bytes:
    """Download and decompress a single ZIP entry via one HTTP Range request."""
    # Local header is 30 bytes + filename + extra (generous 256-byte margin)
    fetch_end = entry["offset"] + 30 + len(entry["name"].encode("utf-8")) + 256 + entry["comp_size"]
    blob = http_range(url, entry["offset"], fetch_end)
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local header for {entry['name']}")
    _, _, _, _, _, _, _, _, _, fn_len, ex_len = struct.unpack("<4sHHHHHIIIHH", blob[:30])
    data_start = 30 + fn_len + ex_len
    comp_data = blob[data_start:data_start + entry["comp_size"]]
    if entry["comp_method"] == 0:
        return comp_data
    elif entry["comp_method"] == 8:
        return zlib.decompress(comp_data, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {entry['comp_method']}")


def main():
    n_per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = os.path.join(base_dir, "data", "EuroSAT_MS")
    os.makedirs(out_root, exist_ok=True)

    entries = parse_zip_central_directory(ZENODO_URL)
    by_class = defaultdict(list)
    for e in entries:
        if not e["name"].endswith(".tif"):
            continue
        parts = e["name"].split("/")
        for cls in EUROSAT_CLASSES:
            if cls in parts:
                by_class[cls].append(e)
                break

    total_downloaded = 0
    for cls in EUROSAT_CLASSES:
        cls_dir = os.path.join(out_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # Pick evenly spaced tiles across the class for maximum geographic diversity
        cls_entries = sorted(by_class[cls], key=lambda x: x["name"])
        step = max(1, len(cls_entries) // n_per_class)
        chosen = cls_entries[::step][:n_per_class]
        print(f"Downloading {len(chosen)} real 13-band GeoTIFFs for {cls} (of {len(cls_entries)} available)...")
        for e in chosen:
            fname = os.path.basename(e["name"])
            dst = os.path.join(cls_dir, fname)
            if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                total_downloaded += 1
                continue
            raw = extract_entry(ZENODO_URL, e)
            with open(dst, "wb") as f:
                f.write(raw)
            total_downloaded += 1

    print(f"Done! Downloaded {total_downloaded} real EuroSAT-MS 13-band GeoTIFFs to {out_root}")


if __name__ == "__main__":
    main()
