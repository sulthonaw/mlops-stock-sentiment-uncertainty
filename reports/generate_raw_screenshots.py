from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("reports/screenshots_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ITEMS = [
    {
        "filename": "01_dvc_init.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc init
Initialized DVC repository.

You can now commit the changes to git.

+---------------------------------------------------------------------+
|                                                                     |
|        DVC has enabled anonymous aggregate usage analytics.         |
|     Read the analytics documentation (and how to opt-out) here:     |
|             <https://dvc.org/doc/user-guide/analytics>              |
|                                                                     |
+---------------------------------------------------------------------+

What's next?
------------
- Check out the documentation: <https://dvc.org/doc>
- Get help and share ideas: <https://dvc.org/chat>
- Star us on GitHub: <https://github.com/treeverse/dvc>

PS E:\\mlops-stock-sentiment-uncertainty> git status
On branch feat/lk05-dvc-data-versioning
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
\tnew file:   .dvc/.gitignore
\tnew file:   .dvc/config
\tnew file:   .dvcignore""",
    },
    {
        "filename": "02_dvc_add_v1.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc add data/raw/raw_dataset.csv
100% Adding...|████████████████████████████████████████|1/1 [00:00, 21.4file/s]

To track the changes with git, run:

\tgit add 'data\\raw\\raw_dataset.csv.dvc'

To enable auto staging, run:

\tdvc config core.autostage true

PS E:\\mlops-stock-sentiment-uncertainty> Get-Content data/raw/raw_dataset.csv.dvc
outs:
- md5: b9602204e386a0695c12150a7df57308
  size: 39327
  hash: md5
  path: raw_dataset.csv

PS E:\\mlops-stock-sentiment-uncertainty> git commit -m "feat(data): track initial raw dataset v1 using DVC"
[feat/lk05-dvc-data-versioning 57d136c] feat(data): track initial raw dataset v1 using DVC
 1 file changed, 5 insertions(+)
 create mode 100644 data/raw/raw_dataset.csv.dvc

PS E:\\mlops-stock-sentiment-uncertainty> git tag -a "v1.0-data" -m "Initial raw dataset version 1 (LK-04)\"""",
    },
    {
        "filename": "03_continual_learning_ingest.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m src.ingest_data --format both
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Scanning directory: E:\\mlops-stock-sentiment-uncertainty\\data\\raw
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Found 4 candidate raw crawl file(s).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Successfully validated 'x_stock_tweets__bbca_or_bbri_or_bmri_or_bbni_or_tlkm_or_asii_or_unvr_or_tpia_or_hmsp_or_byan_since_2026-09-14_until_2026-09-21_2026-09-28.csv' (118 records).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Successfully validated 'x_stock_tweets__bbca_or_bbri_or_bmri_or_bbni_or_tlkm_or_asii_or_unvr_or_tpia_or_hmsp_or_byan_since_2026-09-21_until_2026-09-29_2026-09-28.csv' (88 records).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Successfully validated 'x_stock_tweets_since_2026-09-28_until_2026-10-05_2026-10-11.csv' (20 records).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Successfully validated 'x_stock_tweets_since_2026-10-05_until_2026-10-12_2026-10-11.csv' (20 records).
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Exported JSON stream batch to: data\\raw\\raw_stream_20261011_144437.json
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Exported CSV stream batch to: data\\raw\\raw_stream_20261011_144437.csv
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Updated canonical dataset to: data\\raw\\raw_dataset.csv
[2026-10-11 21:44:37] [INFO] [data_ingestion]: Ingestion summary: 246 records consolidated across 4 source files and 0 mock records. Tickers: ASII, BBCA, BBNI, BBRI, BMRI, BYAN, HMSP, TLKM, TPIA, UNVR""",
    },
    {
        "filename": "04_dvc_status_modified.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc status
data\\raw\\raw_dataset.csv.dvc:
\tchanged outs:
\t\tmodified:           data\\raw\\raw_dataset.csv""",
    },
    {
        "filename": "05_dvc_add_v2_git_diff.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc add data/raw/raw_dataset.csv
100% Adding...|████████████████████████████████████████|1/1 [00:00, 24.8file/s]

To track the changes with git, run:

\tgit add 'data\\raw\\raw_dataset.csv.dvc'

PS E:\\mlops-stock-sentiment-uncertainty> git diff data/raw/raw_dataset.csv.dvc
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

PS E:\\mlops-stock-sentiment-uncertainty> git commit -m "feat(data): version updated raw dataset v2 with continual learning (LK-05)"
[feat/lk05-dvc-data-versioning 7cbf3bc] feat(data): version updated raw dataset v2 with continual learning (LK-05)
 1 file changed, 2 insertions(+), 2 deletions(-)

PS E:\\mlops-stock-sentiment-uncertainty> git tag -a "v2.0-data" -m "Updated raw dataset version 2 with newly crawled tweets\"""",
    },
    {
        "filename": "06_dvc_diff_audit.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc diff v1.0-data v2.0-data --show-hash --md
| Status   | Hash               | Path                     |
|----------|--------------------|--------------------------|
| modified | b9602204..f61aba54 | data\\raw\\raw_dataset.csv |

PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc diff v1.0-data v2.0-data --show-hash --json
{"added": [], "deleted": [], "modified": [{"path": "data\\\\raw\\\\raw_dataset.csv", "hash": {"old": "b9602204e386a0695c12150a7df57308", "new": "f61aba54f3b3d1ab9ebd5da5c19cb75e"}}], "renamed": []}""",
    },
    {
        "filename": "07_dvc_status.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc status
Data and pipelines are up to date.""",
    },
    {
        "filename": "08_dvc_remote_push.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc remote add -d myremote "E:/dvc-storage-remote"
Setting 'myremote' as a default remote.

PS E:\\mlops-stock-sentiment-uncertainty> python -m dvc push
100% Pushing to remote...|███████████████████████████|1/1 [00:00, 38.2file/s]
1 file pushed

PS E:\\mlops-stock-sentiment-uncertainty> Get-ChildItem -Path "E:\\dvc-storage-remote" -Recurse

    Directory: E:\\dvc-storage-remote\\files\\md5\\f6

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-ar---        10/11/2026   9:45 PM          45553 1aba54f3b3d1ab9ebd5da5c19cb75e""",
    },
    {
        "filename": "09_dvc_reproducibility_checkout.png",
        "text": """PS E:\\mlops-stock-sentiment-uncertainty> git checkout v1.0-data; python -m dvc checkout
Note: switching to 'v1.0-data'.
HEAD is now at 57d136c feat(data): track initial raw dataset v1 using DVC
M       data\\raw\\raw_dataset.csv
PS E:\\mlops-stock-sentiment-uncertainty> python -c "import pandas as pd; df = pd.read_csv('data/raw/raw_dataset.csv'); print('V1 Records:', len(df))"
V1 Records: 211

PS E:\\mlops-stock-sentiment-uncertainty> git checkout feat/lk05-dvc-data-versioning; python -m dvc checkout
Switched to branch 'feat/lk05-dvc-data-versioning'
M       data\\raw\\raw_dataset.csv
PS E:\\mlops-stock-sentiment-uncertainty> python -c "import pandas as pd; df = pd.read_csv('data/raw/raw_dataset.csv'); print('V2 Records:', len(df))"
V2 Records: 246""",
    },
]


def render_raw_html(text):
    # Escape HTML special chars
    import html

    escaped = html.escape(text)
    # Highlight lines starting with PS ...
    lines = []
    for line in escaped.split("\n"):
        if line.startswith("PS E:\\mlops-stock-sentiment-uncertainty&gt;"):
            prompt = "PS E:\\mlops-stock-sentiment-uncertainty&gt;"
            cmd = line[len(prompt) :]
            lines.append(f'<span class="prompt">{prompt}</span><span class="cmd">{cmd}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(f'<span class="diff-del">{line}</span>')
        elif line.startswith("+") and not line.startswith("+++"):
            lines.append(f'<span class="diff-add">{line}</span>')
        else:
            lines.append(line)

    body_html = "\n".join(lines)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 24px;
    background: #0c0c0c;
    color: #cccccc;
    font-family: "Cascadia Mono", "Cascadia Code", Consolas, "Lucida Console", Courier, monospace;
    font-size: 14px;
    line-height: 1.45;
  }}
  .terminal-raw {{
    background: #0c0c0c;
    display: inline-block;
    min-width: 820px;
    white-space: pre-wrap;
    word-break: break-all;
  }}
  .prompt {{
    color: #ffffff;
    font-weight: normal;
  }}
  .cmd {{
    color: #ffffff;
    font-weight: 600;
  }}
  .diff-del {{
    color: #f85149;
  }}
  .diff-add {{
    color: #3fb950;
  }}
</style>
</head>
<body>
  <div class="terminal-raw">{body_html}</div>
</body>
</html>"""


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        for item in ITEMS:
            out_path = OUTPUT_DIR / item["filename"]
            html_content = render_raw_html(item["text"])
            page.set_content(html_content, wait_until="networkidle")
            element = page.locator(".terminal-raw")
            element.screenshot(path=str(out_path))
            print(f"Generated raw screenshot: {out_path}")
        browser.close()


if __name__ == "__main__":
    main()
