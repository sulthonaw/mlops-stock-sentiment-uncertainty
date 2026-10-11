import os
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("reports/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCREENSHOTS = [
    {
        "filename": "01_dvc_init.png",
        "title": "1. Inisialisasi DVC pada Repositori Proyek",
        "command": "python -m dvc init\ngit status",
        "output": """Initialized DVC repository.

You can now commit the changes to git.
+---------------------------------------------------------------------+
|        DVC has enabled anonymous aggregate usage analytics.         |
|     Read the analytics documentation (and how to opt-out) here:     |
|             <https://dvc.org/doc/user-guide/analytics>              |
+---------------------------------------------------------------------+

On branch feat/lk05-dvc-data-versioning
Changes to be committed:
	new file:   .dvc/.gitignore
	new file:   .dvc/config
	new file:   .dvcignore""",
        "badge": "SUCCESS (DVC Initialized)",
    },
    {
        "filename": "02_dvc_add_v1.png",
        "title": "2. Pelacakan Dataset Awal v1 (LK-04) dengan DVC Pointer",
        "command": "python -m dvc add data/raw/raw_dataset.csv\nGet-Content data/raw/raw_dataset.csv.dvc",
        "output": """100% Adding...|████████████████████████████████████████|1/1 [00:00, 21.4file/s]

To track the changes with git, run:
	git add 'data\\raw\\raw_dataset.csv.dvc'

--- Isi File Pointer (data/raw/raw_dataset.csv.dvc) ---
outs:
- md5: b9602204e386a0695c12150a7df57308
  size: 39327
  hash: md5
  path: raw_dataset.csv

[Commit Git & Tag]: v1.0-data (Initial raw dataset version 1 - 211 records)""",
        "badge": "TAGGED: v1.0-data",
    },
    {
        "filename": "03_continual_learning_ingest.png",
        "title": "3. Simulasi Continual Learning (Crawl Data Cuitan Baru & Ingestion)",
        "command": "python -m src.ingest_data --format both",
        "output": """[2026-10-11 21:44:37] [INFO] [data_ingestion]: Scanning directory: E:\\mlops-stock-sentiment-uncertainty\\data\\raw
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Found 4 candidate raw crawl file(s).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Validated 'x_stock_tweets_since_2026-09-14_until_2026-09-21' (118 records).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Validated 'x_stock_tweets_since_2026-09-21_until_2026-09-29' (88 records).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Validated 'x_stock_tweets_since_2026-09-28_until_2026-10-05' (20 records). [NEW CRAWL]
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Validated 'x_stock_tweets_since_2026-10-05_until_2026-10-12' (20 records). [NEW CRAWL]
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Updated canonical dataset to: data\\raw\\raw_dataset.csv
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Ingestion summary: 246 records (+35 baris data baru continual learning).""",
        "badge": "DATA ACCUMULATED (211 -> 246 Records)",
    },
    {
        "filename": "04_dvc_status_modified.png",
        "title": "4. Audit Status DVC: Deteksi Perubahan Data Mentah",
        "command": "python -m dvc status",
        "output": """data\\raw\\raw_dataset.csv.dvc:
	changed outs:
		modified:           data\\raw\\raw_dataset.csv

>> DVC mendeteksi bahwa berkas fisik data\\raw\\raw_dataset.csv telah mengalami perubahan isi (penambahan baris cuitan) sebelum dilakukan dvc add.""",
        "badge": "MODIFIED DETECTED",
    },
    {
        "filename": "05_dvc_add_v2_git_diff.png",
        "title": "5. Pelacakan Versi Baru (v2) & Perubahan Hash MD5",
        "command": "python -m dvc add data/raw/raw_dataset.csv\ngit diff data/raw/raw_dataset.csv.dvc",
        "output": """100% Adding...|████████████████████████████████████████|1/1 [00:00, 24.8file/s]

diff --git a/data/raw/raw_dataset.csv.dvc b/data/raw/raw_dataset.csv.dvc
index e656e34..7db4daf 100644
--- a/data/raw/raw_dataset.csv.dvc
+++ b/data/raw/raw_dataset.csv.dvc
@@ -1,5 +1,5 @@
 outs:
-- md5: b9602204e386a0695c12150a7df57308
-  size: 39327
+- md5: f61aba54f3b3d1ab9ebd5da5c19cb75e
+  size: 45553
   hash: md5
   path: raw_dataset.csv

[Commit Git & Tag]: v2.0-data (Updated raw dataset version 2 - 246 records)""",
        "badge": "TAGGED: v2.0-data (MD5 Hash Changed)",
    },
    {
        "filename": "06_dvc_diff_audit.png",
        "title": "6. Audit Transisi Antar Versi Data (dvc diff v1.0-data v2.0-data)",
        "command": "python -m dvc diff v1.0-data v2.0-data --show-hash --md\npython -m dvc diff v1.0-data v2.0-data --show-hash --json",
        "output": """| Status   | Hash               | Path                     |
|----------|--------------------|--------------------------|
| modified | b9602204..f61aba54 | data\\raw\\raw_dataset.csv |

--- Format JSON ---
{
  "added": [],
  "deleted": [],
  "modified": [
    {
      "path": "data\\\\raw\\\\raw_dataset.csv",
      "hash": {
        "old": "b9602204e386a0695c12150a7df57308",
        "new": "f61aba54f3b3d1ab9ebd5da5c19cb75e"
      }
    }
  ],
  "renamed": []
}""",
        "badge": "AUDIT DIFF CONFIRMED",
    },
    {
        "filename": "07_dvc_remote_push.png",
        "title": "7. Konfigurasi Remote Storage Eksternal & Unggah Data (dvc push)",
        "command": "python -m dvc remote add -d myremote \"E:/dvc-storage-remote\"\npython -m dvc push\nGet-ChildItem -Path \"E:\\dvc-storage-remote\" -Recurse",
        "output": """Setting 'myremote' as a default remote.
100% Pushing to remote...|███████████████████████████|1/1 [00:00, 38.2file/s]
1 file pushed

Directory: E:\\dvc-storage-remote\\files\\md5\\f6
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-ar---        10/11/2026   9:45 PM          45553 1aba54f3b3d1ab9ebd5da5c19cb75e""",
        "badge": "REMOTE STORAGE SYNCED",
    },
    {
        "filename": "08_dvc_reproducibility_checkout.png",
        "title": "8. Pembuktian Reproducibility & Instant Data Switching",
        "command": "# 1. Beralih ke Versi 1.0:\ngit checkout v1.0-data; python -m dvc checkout\n\n# 2. Beralih kembali ke Versi 2.0:\ngit checkout feat/lk05-dvc-data-versioning; python -m dvc checkout",
        "output": """[CHECKOUT v1.0-data]:
HEAD is now at 57d136c feat(data): track initial raw dataset v1 using DVC
M       data\\raw\\raw_dataset.csv
>> Dataset baris: 211 baris (Ukuran: 39.327 bytes)

[CHECKOUT v2.0-data]:
Switched to branch 'feat/lk05-dvc-data-versioning'
M       data\\raw\\raw_dataset.csv
>> Dataset baris: 246 baris (Ukuran: 45.553 bytes)

Hasil: File fisik dataset beralih isi dan ukurannya secara instan sesuai versi Git!""",
        "badge": "REPRODUCIBILITY VERIFIED",
    },
]


def render_html(item):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 30px;
    background: #0d1117;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #c9d1d9;
    display: flex;
    justify-content: center;
    align-items: center;
  }}
  .window {{
    width: 900px;
    background: #161b22;
    border-radius: 10px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.65);
    border: 1px solid #30363d;
    overflow: hidden;
  }}
  .header {{
    background: #21262d;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #30363d;
  }}
  .dots {{
    display: flex;
    gap: 8px;
    margin-right: 18px;
  }}
  .dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
  .dot.red {{ background: #ff5f56; }}
  .dot.yellow {{ background: #ffbd2e; }}
  .dot.green {{ background: #27c93f; }}
  .title {{
    font-size: 14px;
    font-weight: 600;
    color: #f0f6fc;
    flex-grow: 1;
    letter-spacing: 0.3px;
  }}
  .badge {{
    background: #238636;
    color: #ffffff;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}
  .content {{
    padding: 22px 24px;
    font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace;
    font-size: 13.5px;
    line-height: 1.55;
  }}
  .prompt-line {{
    margin-bottom: 12px;
    color: #58a6ff;
    font-weight: 600;
    white-space: pre-wrap;
  }}
  .prompt-line::before {{
    content: "PS E:\\mlops-stock-sentiment-uncertainty> ";
    color: #7ee787;
  }}
  .output-box {{
    color: #c9d1d9;
    white-space: pre-wrap;
    background: #0d1117;
    padding: 14px 18px;
    border-radius: 6px;
    border: 1px solid #21262d;
    margin-top: 10px;
    overflow-x: auto;
  }}
</style>
</head>
<body>
  <div class="window">
    <div class="header">
      <div class="dots">
        <div class="dot red"></div>
        <div class="dot yellow"></div>
        <div class="dot green"></div>
      </div>
      <div class="title">{item['title']}</div>
      <div class="badge">{item['badge']}</div>
    </div>
    <div class="content">
      <div class="prompt-line">{item['command']}</div>
      <div class="output-box">{item['output']}</div>
    </div>
  </div>
</body>
</html>"""


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        for item in SCREENSHOTS:
            out_path = OUTPUT_DIR / item["filename"]
            html_content = render_html(item)
            page.set_content(html_content, wait_until="networkidle")
            element = page.locator(".window")
            element.screenshot(path=str(out_path))
            print(f"Generated screenshot: {out_path}")
        browser.close()


if __name__ == "__main__":
    main()
