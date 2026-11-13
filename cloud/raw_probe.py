import gzip
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from urllib.parse import urljoin
from urllib.request import urlopen

work = Path(os.environ["RECSYS_WORK"])
results = Path(os.environ["RECSYS_RESULTS"])
summary = {}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href")


def first_lines(url, count):
    with urlopen(url, timeout=120) as response:
        stream = gzip.GzipFile(fileobj=response) if url.endswith(".gz") else response
        return b"".join(stream.readline() for _ in range(count))


try:
    for year in (2014, 2018, 2023):
        case = work / str(year)
        case.mkdir()
        try:
            if year == 2018:
                metadata = b"".join(json.dumps({"asin": f"i{i}", "title": f"Probe item {i}"}).encode() + b"\n" for i in range(5))
                reviews = b"".join(json.dumps({"asin": f"i{i}", "reviewerID": f"u{u}",
                    "unixReviewTime": 1500000000 + u * 100 + i, "overall": 5}).encode() + b"\n"
                    for u in range(5) for i in range(5))
                provenance = {"input_kind": "synthetic_5x5_export_regression"}
            else:
                if year == 2014:
                    page = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html"
                    with urlopen(page, timeout=120) as response:
                        parser = Links()
                        parser.feed(response.read().decode())
                    review_url = next(urljoin(page, url) for url in parser.urls if url.endswith("/reviews_Office_Products_5.json.gz"))
                    metadata_url = next(urljoin(page, url) for url in parser.urls if url.endswith("/meta_Office_Products.json.gz"))
                else:
                    base = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e/raw/"
                    review_url = base + "review_categories/Industrial_and_Scientific.jsonl"
                    metadata_url = base + "meta_categories/meta_Industrial_and_Scientific.jsonl"
                reviews = first_lines(review_url, 5)
                metadata = first_lines(metadata_url, 2)
                provenance = {"input_kind": "real_source_format_probe", "review_url": review_url,
                              "metadata_url": metadata_url, "review_lines": 5, "metadata_lines": 2}
            (case / "reviews.jsonl").write_bytes(reviews)
            (case / "metadata.jsonl").write_bytes(metadata)
            output = case / "output"
            command = [sys.executable, "minionerec_engine/data/amazon18_data_process.py",
                "--dataset", f"Amazon{year}", "--reviews_file", str(case / "reviews.jsonl"),
                "--metadata_file", str(case / "metadata.jsonl"), "--output_path", str(output),
                "--st_year", "1996", "--ed_year", "2025", "--ed_month", "1"]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
            log = completed.stdout + completed.stderr
            (results / f"preprocessing-{year}.log").write_text(log)
            files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
            summary[str(year)] = provenance | {"returncode": completed.returncode,
                "output_files": files, "reported_success": "Preprocessing completed" in completed.stdout,
                "review_sample_sha256": hashlib.sha256(reviews).hexdigest(),
                "metadata_sample_sha256": hashlib.sha256(metadata).hexdigest(), "log_tail": log[-3500:]}
        except Exception:
            summary[str(year)] = {"probe_error": traceback.format_exc()}
        print("RAW_CASE", year, json.dumps(summary[str(year)]), flush=True)
finally:
    (results / "raw-formats.json").write_text(json.dumps(summary, indent=2))
    print("RAW_SUMMARY", json.dumps(summary), flush=True)
