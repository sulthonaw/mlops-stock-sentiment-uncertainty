# MLOps: Penerapan MLOps Berkelanjutan untuk Analisis Sentimen Saham Big Cap Indonesia Menggunakan Uncertainty Sampling dan Data Dinamis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Informasi Mahasiswa
- **Nama:** Muhammad Sulthon Aulia Wijaya
- **NIM:** 245150207111103
- **Kelas:** TIF-A
- **Mata Kuliah:** MLOps
- **Program Studi:** Teknik Informatika, Fakultas Ilmu Komputer, Universitas Brawijaya (2026)

---

## 📖 Penjelasan Proyek
Proyek ini mengimplementasikan sistem **MLOps berkelanjutan (*Continuous Training & Active Learning*)** untuk klasifikasi sentimen (*positive*, *neutral*, *negative*) serta *Aspect-Based Sentiment Analysis* pada percakapan media sosial (X/Twitter) terkait 10 saham Big Cap di Bursa Efek Indonesia (IDX):
> **Daftar Saham:** `BBCA`, `BBRI`, `BMRI`, `BBNI`, `TLKM`, `ASII`, `UNVR`, `TPIA`, `HMSP`, `BYAN`.

### 📊 Dataset yang Digunakan
1. **Baseline Dataset:** **ID-SMSA (*Indonesian Stock Market Dataset for Sentiment Analysis*)**
   - **Sumber:** Mendeley Data (DOI: `10.17632/tn4vzs8tdw.3`)
   - **Jumlah:** 3.288 cuitan teranotasi (1.769 *Positive*, 733 *Neutral*, 786 *Negative*) periode 2021–2024.
   - **Fungsi:** Data awal untuk melatih model baseline IndoBERT v1.0.
2. **Data Dinamis (Cuitan Terbaru):**
   - **Metode:** Pengumpulan data harian secara mandiri melalui *Custom Browser Extension* (DOM parsing X Web) tanpa dependensi API X berbayar.
   - **Fungsi:** Sumber data dinamis untuk mendeteksi pergeseran pola bahasa (*data/concept drift*) dan pengayaan *Active Learning*.

### 💡 Pendekatan & Solusi MLOps
- **Tantangan:** Bahasa percakapan pasar modal sangat dinamis (*slang*, istilah *cuan, cut loss, ARA, ARB, window dressing*). Model tanpa pembaruan berkala akan mengalami penurunan performa akibat *concept drift* dan *covariate shift*.
- **Solusi Pipeline:**
  - **Baseline Training:** Fine-tuning IndoBERT menggunakan dataset ID-SMSA dan pelacakan eksperimen via MLflow.
  - **Continuous Ingestion:** Penyerapan data cuitan harian melalui FastAPI dan versioning dataset menggunakan DVC.
  - **Uncertainty Sampling & Retraining Loop:**
    - *Confidence > 85%* $\rightarrow$ Hasil prediksi disimpan dan ditampilkan pada dashboard.
    - *Confidence 40% – 60%* $\rightarrow$ Ditandai dan dikumpulkan ke **Active Learning Pool**.
    - *Pemicu Retraining* $\rightarrow$ Retraining model terpicu otomatis saat pool mencapai 100 data atau evaluasi berkala Macro F1-Score $< 0.75$.

---

## 🏗️ Arsitektur Pipeline

```mermaid
flowchart TD
    subgraph L1["Layer 1: Data Ingestion & Acquisition"]
        A1["Baseline: ID-SMSA (3.288 Cuitan)"]
        A2["X Web Search Query (10 Big Cap)"]
        A3["Custom Browser Extension Scraper"]
        A4["FastAPI Ingestion Endpoint"]
        A2 --> A3 --> A4
    end

    subgraph L2["Layer 2: Data Preprocessing & Versioning"]
        B1["Text Cleaning ([USERNAME], [URL], [HASHTAG])"]
        B2["DVC Data Versioning & Storage"]
        A1 --> B1
        A4 --> B1
        B1 --> B2
    end

    subgraph L3["Layer 3: Model Experimentation & Registry"]
        C1["IndoBERT Fine-Tuning"]
        C2["MLflow Experiment Tracking"]
        C3["Validation & Model Evaluation (Macro F1)"]
        C4["MLflow Model Registry (Production Model)"]
        B2 --> C1
        C1 <--> C2
        C1 --> C3 --> C4
    end

    subgraph L4["Layer 4: Serving, Monitoring & Active Learning Loop"]
        D1["REST API Serving (Docker / FastAPI)"]
        D2["Real-Time Sentiment Dashboard"]
        D3["Confidence Score Monitoring"]
        D4["Uncertainty Sampling (40-60% Confidence)"]
        D5["Active Learning Pool (Threshold: 100 data)"]
        D6["Human-in-the-Loop Annotation"]

        C4 --> D1
        D1 --> D2
        D1 --> D3
        D3 -- "Confidence > 85%" --> D2
        D3 -- "Confidence 40-60%" --> D4
        D4 --> D5 --> D6
        D6 -->|"Retraining Trigger"| C1
    end
```

---

## 📂 Struktur Direktori

```text
mlops-stock-sentiment-uncertainty/
├── .devcontainer/
│   └── devcontainer.json        # Konfigurasi container otomatis GitHub Codespaces
├── config/
│   ├── config.yaml              # Hyperparameter, path, dan ambang batas retraining
│   └── .gitkeep
├── data/
│   ├── .gitignore               # Aturan ignorasi data (unignore crawl CSV & raw stream)
│   ├── raw/
│   │   ├── .gitkeep             # Dataset baseline ID-SMSA
│   │   ├── x_stock_tweets_*.csv # Cuitan mentah hasil crawl extension 10 saham Big Cap
│   │   └── raw_stream_*.json    # Batch konsolidasi stempel waktu & metadata
│   ├── interim/
│   │   ├── .gitkeep
│   │   └── interim_stream_*.parquet # Data bersih standar ID-SMSA (Parquet/CSV)
│   ├── processed/
│   │   └── .gitkeep             # Data siap training & active learning pool
│   └── external/
│       └── .gitkeep             # Kamus slang/lexicon finansial eksternal
├── models/
│   ├── .gitignore               # Menjaga bobot model biner (.pt, .bin) tidak ter-commit
│   └── .gitkeep                 # Folder checkpoint model terlatih
├── notebooks/
│   ├── 01_initial_eda.ipynb     # Notebook eksplorasi data baseline ID-SMSA
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── ingest_data.py           # Penarikan, validasi, & konsolidasi data dinamis
│   ├── preprocess.py            # Automasi normalisasi teks standar ID-SMSA
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py               # REST API FastAPI untuk data ingestion & inferensi
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py      # Modul pembersihan & pra-pemrosesan data
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # Modul tokenisasi & feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py       # Pipeline pelatihan & tracking model
│   │   └── predict_model.py     # Logika inferensi & evaluasi ketidakpastian
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Utility logging standar
├── .gitignore                   # Aturan ignorasi Python, Venv, IDE, MLflow, OS
├── LICENSE                      # Lisensi MIT (2026)
├── requirements.txt             # Daftar dependensi library Python
└── README.md                    # Dokumentasi utama proyek
```

---

## 🚀 Panduan Menjalankan Proyek

### Opsi 1: Menjalankan di GitHub Codespaces (Rekomendasi)
Repositori ini telah dikonfigurasi dengan `.devcontainer/devcontainer.json` sehingga seluruh lingkungan kerja (Python 3.11, dependensi library, dan ekstensi VS Code) akan terpasang secara otomatis:

1. Klik tombol **Code** (berwarna hijau) di halaman utama repositori GitHub.
2. Pilih tab **Codespaces** lalu klik **Create codespace on main**.
3. Tunggu container selesai dibangun. Lingkungan kerja langsung siap digunakan.

### Opsi 2: Menjalankan Secara Lokal

1. **Clone repositori:**
   ```bash
   git clone https://github.com/sulthonaw/mlops-stock-sentiment-uncertainty.git
   cd mlops-stock-sentiment-uncertainty
   ```

2. **Buat dan aktifkan Virtual Environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install seluruh dependensi:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📥 Implementasi LK-04: Penarikan Data Dinamis & Automasi Prapemrosesan

Pada tahapan **LK-04**, pipeline MLOps ini telah dilengkapi dengan komponen penarikan data (*data ingestion*) dinamis dan automasi prapemrosesan (*text preprocessing automation*) untuk mengolah percakapan saham mentah dari platform X secara terstruktur dan non-destruktif.

### 1. Format Data Mentah Hasil Web Crawling
Data mentah dikumpulkan menggunakan *custom browser extension* tanpa biaya API berbayar dengan menyasar 10 emiten saham Big Cap Indonesia:
`$BBCA`, `$BBRI`, `$BMRI`, `$BBNI`, `$TLKM`, `$ASII`, `$UNVR`, `$TPIA`, `$HMSP`, `$BYAN`.

Berkas hasil crawl disimpan langsung pada `data/raw/x_stock_tweets_*.csv` dengan skema yang telah diselaraskan dengan baseline ID-SMSA:
- `Sentence`: Teks cuitan mentah pengguna.
- `Sentiment`: Label sentimen (`null` / *unlabeled* untuk data dinamis).
- `Tweet Date`: Waktu cuitan dibuat (format string Twitter).
- `English Translation`: Terjemahan bahasa Inggris (opsional).
- `Favorite Count`: Jumlah suka (*like*).
- `Retweet Count`: Jumlah cuitan ulang (*retweet*).
- `Reply Count`: Jumlah balasan (*reply*).
- `Quote Count`: Jumlah kutipan (*quote*).

### 2. Skrip Ingestion Dinamis (`src/ingest_data.py`)
Skrip ini bertugas memindai, memvalidasi skema, mendeteksi emiten, dan mengonsolidasi berkas crawl ke dalam format batch stream berstempel waktu tanpa menimpa (*non-destructive*) file lama.

**Fitur Utama:**
- **Validasi Skema Otomatis:** Memastikan keberadaan dan urutan 8 kolom standar.
- **Konsolidasi Berstempel Waktu:** Menghasilkan berkas baru `data/raw/raw_stream_%Y%m%d_%H%M%S.json` dan `.csv`.
- **Batch Metadata Tracking:** Mencatat `ingestion_timestamp`, `batch_id`, `total_records`, daftar emiten (`tickers`), `schema_version`, dan file sumber.
- **Simulasi Live Stream (`--mock-new`):** Menyediakan opsi simulasi penambahan data cuitan baru terkini untuk pengujian berkala.

**Contoh Eksekusi CLI Ingestion:**
```bash
# 1. Ingest seluruh berkas crawl mentah (ekspor JSON dan CSV)
python src/ingest_data.py --format both

# 2. Ingest berkas crawl mentah sekaligus menyimulasikan data stream baru
python src/ingest_data.py --mock-new --mock-count 10 --format both
```

### 3. Automasi Prapemrosesan Data (`src/preprocess.py`)
Skrip ini mentransformasi cuitan mentah dari `data/raw/` menjadi korpus bersih terstandarisasi di `data/interim/`.

**Tahapan Normalisasi (Konsisten Standar ID-SMSA):**
1. **Normalisasi Akun (`@username`):** Diganti menjadi token `[USERNAME]`.
2. **Normalisasi Tautan (URL):** Diganti menjadi token `[URL]`.
3. **Normalisasi Tagar (`#hashtag`):** Diganti menjadi token `[HASHTAG]`.
4. **Pembersihan Whitespace:** Menghilangkan baris baru berlebih (`\n\r`) dan mereduksi spasi ganda menjadi spasi tunggal.
5. **Penanganan Missing Values:** Menghapus baris dengan teks kosong atau `NaN`.
6. **Deduplikasi Teks:** Menghapus entri cuitan duplikat berdasarkan kolom `Sentence`.
7. **Penyimpanan Berkas Interim:** Disimpan ke `data/interim/interim_stream_<source>_<timestamp>.parquet` dan `.csv` disertai berkas metadata pelacak.

**Contoh Eksekusi CLI Prapemrosesan:**
```bash
# 1. Prapemrosesan batch stream terbaru otomatis dari data/raw/
python src/preprocess.py --format both

# 2. Prapemrosesan berkas batch mentah spesifik
python src/preprocess.py --input-path data/raw/raw_stream_20260928_144339.json --format both
```

**Ringkasan Log Terminal yang Dihasilkan:**
```text
============================================================
           RINGKASAN PRAPEMROSESAN DATA (LK-04)
============================================================
Berkas Sumber        : raw_stream_20260928_144339.json
Jumlah Baris Awal    : 211
Missing Values Dibuang: 0
Duplikat Dihapus     : 66
Total Baris Bersih   : 145
Format Ekspor        : both
Direktori Output     : E:\mlops-stock-sentiment-uncertainty\data\interim
============================================================
```

---

## 🔄 Pelacakan & Versioning Dataset Menggunakan DVC (LK-05)

Pada tahapan **LK-05**, sistem MLOps diintegrasikan dengan **DVC (*Data Version Control*)** untuk melacak evolusi dataset mentah secara independen dari Git. Pendekatan ini memastikan Git hanya menyimpan berkas metadata pointer berukuran kecil (`.dvc`), sementara artefak data berukuran besar dikelola oleh DVC dengan dukungan *remote storage* dan reproduktibilitas penuh (*continual learning*).

Selain itu, modul **Headless Playwright Crawler** diintegrasikan langsung ke dalam repositori (`src/crawler/`) menggantikan metode ekstensi browser sebelumnya untuk memastikan otomatisasi penuh (*fully automated scraping*) sesuai masukan dosen.

### 📐 Arsitektur Siklus Hidup Data (Crawler + DVC + Git)

```mermaid
flowchart TD
    subgraph DataAcquisition["1. Otomasi Perolehan Data (Playwright Stealth Crawler)"]
        C1["X (Twitter) Live / Top Search"]
        C2["src/crawler/crawler.py<br/>(Playwright Headless + Cookie Auth)"]
        C3["Rentang Tanggal (--since, --until)<br/>Minggu Kemarin & Minggu Ini"]
        C1 --> C2 --> C3
    end

    subgraph DynamicIngestion["2. Ingesti & Konsolidasi Dinamis"]
        I1["src/ingest_data.py"]
        I2["data/raw/raw_dataset.csv (Canonical)"]
        I3["data/raw/raw_stream_*.csv & .json"]
        C3 --> I1
        I1 --> I2
        I1 --> I3
    end

    subgraph DVCVersioning["3. DVC Tracking & Git Synchronization"]
        D1["dvc add data/raw/raw_dataset.csv"]
        D2["data/raw/raw_dataset.csv.dvc (MD5 Pointer)"]
        D3[".dvc/cache/ & Remote Storage"]
        G1["Git Commit & Tags (v1.0-data, v2.0-data)"]
        I2 --> D1
        D1 --> D2
        D1 --> D3
        D2 --> G1
    end

    subgraph AuditDiff["4. Audit, Diff & Reproducibility"]
        A1["dvc diff v1.0-data v2.0-data"]
        A2["dvc status"]
        A3["dvc checkout (Instant Version Switching)"]
        G1 --> A1
        G1 --> A2
        G1 --> A3
    end
```

---

### 🚀 1. Modul Automated Stealth Crawler (`src/crawler/`)

Crawler otomatis berbasis **Playwright Python** yang mampu melakukan ekstraksi cuitan saham tanpa dependensi berbayar:
- **Teknik Stealth:** Argument `--disable-blink-features=AutomationControlled` dan sanitasi `navigator.webdriver`.
- **Injeksi Session Cookies:** Menggunakan `auth_token` dan `ct0` dari file konfigurasi `.env`.
- **Anti Virtual DOM Recycling:** Menggunakan instant *Hash Set deduplication* (`seen_tweet_ids`) sebelum elemen cuitan di-*unmount* oleh browser.
- **Dukungan Rentang Waktu Dinamis:** Mendukung argumen `--since` dan `--until` untuk memfilter data per minggu (7 hari).

**Contoh Eksekusi Crawler:**
```bash
# 1. Crawl data minggu kemarin (28 Sep - 05 Okt 2026, 20 cuitan)
python -m src.crawler.crawler --since 2026-09-28 --until 2026-10-05 --max 20 --headless

# 2. Crawl data minggu ini (05 Okt - 12 Okt 2026, 20 cuitan)
python -m src.crawler.crawler --since 2026-10-05 --until 2026-10-12 --max 20 --headless

# 3. Eksekusi via ingest_data langsung:
python -m src.ingest_data --crawl --crawl-since 2026-10-05 --crawl-until 2026-10-12 --crawl-max 20
```

---

### 📦 2. Alur Inisialisasi & Versioning Data DVC

#### Langkah 1: Inisialisasi DVC
Inisialisasi DVC di repositori proyek:
```bash
dvc init
git commit -m "chore(dvc): initialize DVC repository"
```
Perintah ini menghasilkan konfigurasi `.dvc/` dan file `.dvcignore`.

#### Langkah 2: Pelacakan Dataset Awal (Versi 1.0 - LK-04)
Dataset awal hasil konsolidasi LK-04 (211 baris) dilacak sebagai versi awal:
```bash
# Lacak dataset kanonikal ke DVC
dvc add data/raw/raw_dataset.csv

# Simpan metadata pointer ke Git dan beri Tag rilis
git add data/raw/raw_dataset.csv.dvc
git commit -m "feat(data): track initial raw dataset v1 using DVC"
git tag -a "v1.0-data" -m "Initial raw dataset version 1 (LK-04)"
```
**Metadata Pointer Versi 1.0 (`data/raw/raw_dataset.csv.dvc`):**
```yaml
outs:
- md5: b9602204e386a0695c12150a7df57308
  size: 39327
  hash: md5
  path: raw_dataset.csv
```

#### Langkah 3: Simulasi Continual Learning (Penambahan Data Cuitan Baru)
Menjalankan crawling batch minggu kemarin & minggu ini, kemudian mengonsolidasi seluruh cuitan ke dataset utama:
```bash
python -m src.ingest_data --format both
```
Dataset `data/raw/raw_dataset.csv` bertambah dari **211 baris** menjadi **246 baris** (+35 baris data baru).

#### Langkah 4: Versioning Dataset Baru (Versi 2.0 - LK-05)
Lacak kembali perubahan dataset dengan DVC:
```bash
# Lacak ulang data yang diperbarui
dvc add data/raw/raw_dataset.csv

# Commit perubahan hash pointer ke Git dan beri Tag rilis
git add data/raw/raw_dataset.csv.dvc
git commit -m "feat(data): version updated raw dataset v2 with continual learning (LK-05)"
git tag -a "v2.0-data" -m "Updated raw dataset version 2 with newly crawled tweets"
```
**Perubahan Metadata Pointer (`git diff data/raw/raw_dataset.csv.dvc`):**
```diff
 outs:
-- md5: b9602204e386a0695c12150a7df57308
-  size: 39327
+- md5: f61aba54f3b3d1ab9ebd5da5c19cb75e
+  size: 45553
   hash: md5
   path: raw_dataset.csv
```

---

### 🔍 3. Audit Perubahan Data & Transisi Silsilah (`dvc diff` & `dvc status`)

Untuk memverifikasi perbedaan antar dua versi dataset secara jelas tanpa memuat file biner besar ke Git:

#### A. Pemeriksaan Status (`dvc status`)
```bash
dvc status
```
**Output:**
```text
Data and pipelines are up to date.
```

#### B. Perbandingan Dua Versi Dataset (`dvc diff v1.0-data v2.0-data`)
```bash
dvc diff v1.0-data v2.0-data --show-hash --md
```
**Output Tabel Audit:**
| Status   | Hash               | Path                     |
|----------|--------------------|--------------------------|
| modified | b9602204..f61aba54 | data\raw\raw_dataset.csv |

Format JSON untuk integrasi CI/CD (`dvc diff v1.0-data v2.0-data --show-hash --json`):
```json
{
  "added": [],
  "deleted": [],
  "modified": [
    {
      "path": "data\\raw\\raw_dataset.csv",
      "hash": {
        "old": "b9602204e386a0695c12150a7df57308",
        "new": "f61aba54f3b3d1ab9ebd5da5c19cb75e"
      }
    }
  ],
  "renamed": []
}
```

---

### ☁️ 4. Konfigurasi Remote Storage & Reproduktibilitas

Untuk menjamin dataset tersimpan di luar repositori lokal (*external storage / cloud*):

#### Konfigurasi Remote Storage:
```bash
# Menambahkan remote storage default (lokal / mount path / S3 / MinIO)
dvc remote add -d myremote E:/dvc-storage-remote
git commit -am "chore(dvc): configure remote storage and push dataset"

# Mengunggah data ke remote storage
dvc push
```

#### Pengujian Reproducibility & Instant Data Switching:
```bash
# 1. Beralih ke dataset Versi 1.0 (211 baris)
git checkout v1.0-data
dvc checkout

# 2. Beralih kembali ke dataset Versi 2.0 (246 baris)
git checkout feat/lk05-dvc-data-versioning
dvc checkout
```
Hasil verifikasi menunjukkan bahwa file fisik `data/raw/raw_dataset.csv` langsung beralih isi dan ukurannya secara instan sesuai versi Git yang di-*checkout*.

---

### 📷 5. Berkas Tangkapan Layar (Screenshots untuk Laporan PDF)

Seluruh tangkapan layar terminal telah diekspor dalam format resolusi tinggi di direktori `reports/screenshots/`:
1. `reports/screenshots/01_dvc_init.png` - Inisialisasi DVC & status Git.
2. `reports/screenshots/02_dvc_add_v1.png` - Tracking dataset awal v1 & file pointer `.dvc`.
3. `reports/screenshots/03_continual_learning_ingest.png` - Eksekusi ingesti & akumulasi data cuitan baru.
4. `reports/screenshots/04_dvc_status_modified.png` - Deteksi otomatis modifikasi data oleh DVC status.
5. `reports/screenshots/05_dvc_add_v2_git_diff.png` - Tracking v2 & perbedaan MD5 hash pada git diff.
6. `reports/screenshots/06_dvc_diff_audit.png` - Audit transisi data dengan `dvc diff`.
7. `reports/screenshots/07_dvc_remote_push.png` - Konfigurasi remote storage & `dvc push`.
8. `reports/screenshots/08_dvc_reproducibility_checkout.png` - Bukti beralih data instan dengan `dvc checkout`.

