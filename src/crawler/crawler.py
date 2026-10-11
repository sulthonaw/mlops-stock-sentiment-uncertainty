#!/usr/bin/env python3
"""
Crawler Otomatis X (Twitter) Menggunakan Playwright Python
Fitur:
- Headless Chromium dengan argument anti-deteksi: --disable-blink-features=AutomationControlled
- Injeksi session cookies (auth_token & ct0) via python-dotenv
- Dynamic mouse wheeling dengan jitter (2.0 - 3.5 detik)
- Anti Virtual DOM Recycling dengan instant hash set deduplication
- Ekspor 8 kolom ke CSV (utf-8-sig) menggunakan pandas
"""

import os
import re
import sys
import time
import random
import logging
import argparse
from typing import Optional, Dict, Any, List, Set
from dotenv import load_dotenv
import pandas as pd
from playwright.sync_api import sync_playwright, BrowserContext, Page, Locator

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("X-Crawler")

DEFAULT_TICKERS = [
    "BBCA",
    "BBRI",
    "BMRI",
    "BBNI",
    "TLKM",
    "ASII",
    "UNVR",
    "TPIA",
    "HMSP",
    "BYAN",
]


def build_search_url(
    tickers: Optional[List[str]] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    filter_mode: str = "top",
) -> str:
    """Build URL search query X untuk ticker saham dan rentang waktu."""
    from urllib.parse import quote

    selected_tickers = tickers or DEFAULT_TICKERS
    ticker_query = " OR ".join([f"${t}" for t in selected_tickers])
    query_parts = [f"({ticker_query})"]
    if since_date:
        query_parts.append(f"since:{since_date}")
    if until_date:
        query_parts.append(f"until:{until_date}")

    query_str = " ".join(query_parts)
    encoded_query = quote(query_str)
    return f"https://x.com/search?q={encoded_query}&src=typed_query&f={filter_mode}"



def parse_metric(raw_text: Optional[str]) -> int:
    """
    Mengonversi string metrik angka Twitter menjadi integer.
    Mendukung format bahasa Inggris (1.2K, 3.5M, 150) dan bahasa Indonesia (1,2 rb, 3,5 jt, 1331 Suka).
    """
    if not raw_text:
        return 0

    text = raw_text.strip().replace("\xa0", " ").lower()
    # Mencari pola angka desimal/bulat diikuti optional satuan (k, rb, jt, m, b) dengan boundary kata
    match = re.search(r"([\d\.,]+)\s*(?:(rb|ribu|k|jt|juta|m|b)\b)?", text)
    if not match:
        return 0

    num_str, unit = match.groups()
    if not num_str:
        return 0

    if unit:
        # Jika ada satuan, koma/titik berfungsi sebagai pemisah desimal
        num_str = num_str.replace(",", ".")
        try:
            val = float(num_str)
            if unit in ["k", "rb", "ribu"]:
                return int(val * 1_000)
            elif unit in ["m", "jt", "juta"]:
                return int(val * 1_000_000)
            elif unit in ["b"]:
                return int(val * 1_000_000_000)
            return int(val)
        except (ValueError, TypeError):
            return 0
    else:
        # Tanpa satuan, bersihkan titik atau koma pemisah ribuan (misal: 1,331 atau 1.331)
        clean_num = num_str.replace(",", "").replace(".", "")
        try:
            return int(clean_num)
        except (ValueError, TypeError):
            return 0


def extract_metric_by_testids(container: Locator, testids: List[str]) -> int:
    """
    Mengekstrak angka metrik dari elemen tombol aksi Twitter tanpa terikat tag spesifik.
    Memeriksa atribut aria-label terlebih dahulu (karena berisi angka presisi/unrounded),
    lalu fallback ke inner_text.
    """
    for tid in testids:
        try:
            # Menggunakan selector tag-agnostik [data-testid="..."] (bisa <button>, <div>, dll)
            target = container.locator(f'[data-testid="{tid}"]').first
            if target.count() == 0:
                continue

            # 1. Periksa aria-label terlebih dahulu (contoh: "1331 Suka. Menyukai")
            aria_label = target.get_attribute("aria-label")
            if aria_label:
                val = parse_metric(aria_label)
                if val > 0:
                    return val

            # 2. Periksa inner text tombol jika aria-label kosong atau tidak memuat angka
            text = target.inner_text().strip()
            if text:
                val = parse_metric(text)
                if val > 0:
                    return val
        except Exception:
            continue
    return 0


def extract_tweet_id(article: Locator) -> Optional[str]:
    """
    Mengekstrak Tweet ID unik dari link status tweet (/status/{id}).
    Digunakan untuk deduplikasi pada hash set untuk mengatasi Virtual DOM Recycling.
    """
    try:
        links = article.locator("a[href*='/status/']").all()
        for link in links:
            href = link.get_attribute("href")
            if href:
                match = re.search(r"/status/(\d+)", href)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None


def extract_tweet_data(article: Locator) -> Optional[Dict[str, Any]]:
    """
    Mengekstrak atribut data tweet sesuai skema 8 kolom wajib:
    Sentence, Sentiment, Tweet Date, English Translation, Favorite Count,
    Retweet Count, Reply Count, Quote Count.
    """
    try:
        # 1. Sentence: Teks cuitan asli (div[data-testid="tweetText"])
        tweet_text_locator = article.locator('div[data-testid="tweetText"]').first
        if tweet_text_locator.count() == 0:
            return None
        sentence = tweet_text_locator.inner_text().strip()
        if not sentence:
            return None

        # 2. Sentiment: null (dikosongkan / None)
        sentiment = None

        # 3. Tweet Date: Atribut datetime pada tag <time>
        time_locator = article.locator("time").first
        tweet_date = None
        if time_locator.count() > 0:
            tweet_date = time_locator.get_attribute("datetime")

        # 4. English Translation: Dikosongkan (None)
        english_translation = None

        # 5. Favorite Count: Integer likes (data-testid: like / unlike)
        favorite_count = extract_metric_by_testids(article, ["like", "unlike"])

        # 6. Retweet Count: Integer reposts (data-testid: retweet / unretweet)
        retweet_count = extract_metric_by_testids(article, ["retweet", "unretweet"])

        # 7. Reply Count: Integer balasan (data-testid: reply)
        reply_count = extract_metric_by_testids(article, ["reply"])

        # 8. Quote Count: Integer kuotasi
        quote_count = 0

        # Layer Redundansi: Parse dari div[role="group"] aria-label
        # (Format ID: "89 balasan, 213 posting ulang, 1331 suka, 122 markah, 214095 tayangan")
        # (Format EN: "89 replies, 213 reposts, 1331 likes, 122 bookmarks, 214095 views")
        group = article.locator('div[role="group"]').first
        if group.count() > 0:
            group_aria = group.get_attribute("aria-label") or ""
            if group_aria:
                if favorite_count == 0:
                    fav_match = re.search(r"([\d\.,\s]+(?:rb|ribu|k|jt|juta|m)?)\s*(?:suka|likes?)", group_aria, re.I)
                    if fav_match:
                        favorite_count = parse_metric(fav_match.group(1))

                if retweet_count == 0:
                    rt_match = re.search(r"([\d\.,\s]+(?:rb|ribu|k|jt|juta|m)?)\s*(?:posting ulang|reposts?|retweets?)", group_aria, re.I)
                    if rt_match:
                        retweet_count = parse_metric(rt_match.group(1))

                if reply_count == 0:
                    rep_match = re.search(r"([\d\.,\s]+(?:rb|ribu|k|jt|juta|m)?)\s*(?:balasan|repl(?:y|ies))", group_aria, re.I)
                    if rep_match:
                        reply_count = parse_metric(rep_match.group(1))

                # Deteksi Quote Count dari aria-label grup jika ada
                quote_match = re.search(r"([\d\.,\s]+(?:rb|ribu|k|jt|juta|m)?)\s*(?:kutipan|quotes?)", group_aria, re.I)
                if quote_match:
                    quote_count = parse_metric(quote_match.group(1))

        # Fallback Quote Count dari link kutipan jika ada di DOM tweet
        if quote_count == 0:
            quote_link = article.locator("a[href*='/quotes']").first
            if quote_link.count() > 0:
                quote_text = quote_link.inner_text().strip()
                if quote_text:
                    quote_count = parse_metric(quote_text)

        return {
            "Sentence": sentence,
            "Sentiment": sentiment,
            "Tweet Date": tweet_date,
            "English Translation": english_translation,
            "Favorite Count": favorite_count,
            "Retweet Count": retweet_count,
            "Reply Count": reply_count,
            "Quote Count": quote_count,
        }
    except Exception as e:
        logger.debug(f"Gagal mengekstrak elemen tweet: {e}")
        return None


def inject_session_cookies(context: BrowserContext, auth_token: str, ct0: str) -> None:
    """
    Menginjeksikan session cookies auth_token dan ct0 secara dinamis
    ke context browser untuk domain .x.com dan .twitter.com.
    """
    domains = [".x.com", ".twitter.com"]
    cookies = []

    for domain in domains:
        cookies.extend([
            {
                "name": "auth_token",
                "value": auth_token,
                "domain": domain,
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            },
            {
                "name": "ct0",
                "value": ct0,
                "domain": domain,
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            },
        ])

    context.add_cookies(cookies)
    logger.info("Session cookies (auth_token & ct0) berhasil diinjeksikan.")


def run_crawler(
    target_url: str,
    max_tweets: int = 50,
    headless: bool = True,
    output_file: str = "tweets_output.csv",
) -> None:
    """
    Menjalankan otomasi crawling Playwright dengan argumen anti-deteksi,
    jitter dynamic wheeling, dan deduplikasi instan.
    """
    # 1. Validasi Kredensial Lingkungan
    auth_token = os.getenv("AUTH_TOKEN")
    ct0 = os.getenv("CT0")

    if not auth_token or not ct0 or auth_token == "your_auth_token_here":
        logger.error(
            "Kredensial AUTH_TOKEN dan CT0 wajib diatur di dalam file .env!\n"
            "Silakan salin .env.example menjadi .env lalu isi nilainya."
        )
        sys.exit(1)

    # Hash Set untuk deduplikasi ID Tweet (mengatasi Virtual DOM Recycling)
    seen_tweet_ids: Set[str] = set()
    collected_data: List[Dict[str, Any]] = []

    logger.info(f"Target URL : {target_url}")
    logger.info(f"Max Tweets : {max_tweets}")
    logger.info(f"Mode       : {'Headless' if headless else 'Headed'}")
    logger.info(f"Output     : {output_file}")

    with sync_playwright() as p:
        # Konfigurasi Chromium Stealth
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--start-maximized",
        ]

        browser = p.chromium.launch(
            headless=headless,
            args=launch_args,
        )

        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="Asia/Jakarta",
        )

        # Inject Script untuk menghapus jejak navigator.webdriver
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        # Injeksi Cookies Sesi
        inject_session_cookies(context, auth_token, ct0)

        page: Page = context.new_page()

        try:
            logger.info("Membuka halaman target...")
            page.goto(target_url, wait_until="domcontentloaded", timeout=60000)

            # Tunggu elemen timeline termuat
            logger.info("Menunggu timeline X siap...")
            try:
                page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
            except Exception:
                logger.warning(
                    "Timeout menunggu timeline. Mencoba reload halaman..."
                )
                try:
                    page.reload(wait_until="domcontentloaded", timeout=45000)
                    page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
                except Exception:
                    logger.warning("Melanjutkan observasi DOM langsung...")

            stagnant_scroll_count = 0
            max_stagnant_scrolls = 10

            while len(collected_data) < max_tweets:
                # Ambil semua cuitan yang sedang aktif ter-render di Virtual DOM
                articles = page.locator('article[data-testid="tweet"]').all()
                new_tweets_in_this_step = 0

                for article in articles:
                    tweet_id = extract_tweet_id(article)
                    if not tweet_id:
                        continue

                    # Instant hash set deduplication
                    if tweet_id not in seen_tweet_ids:
                        seen_tweet_ids.add(tweet_id)
                        tweet_data = extract_tweet_data(article)
                        if tweet_data:
                            collected_data.append(tweet_data)
                            new_tweets_in_this_step += 1
                            logger.info(
                                f"[{len(collected_data)}/{max_tweets}] "
                                f"Tweet ID {tweet_id} | "
                                f"Fav: {tweet_data['Favorite Count']} | "
                                f"RT: {tweet_data['Retweet Count']} | "
                                f"Rep: {tweet_data['Reply Count']} | "
                                f"Qte: {tweet_data['Quote Count']} | "
                                f"Snippet: {tweet_data['Sentence'][:35]}..."
                            )

                        if len(collected_data) >= max_tweets:
                            break

                if new_tweets_in_this_step == 0:
                    stagnant_scroll_count += 1
                    logger.debug(
                        f"Tidak ada tweet baru pada scroll ini ({stagnant_scroll_count}/{max_stagnant_scrolls})"
                    )
                else:
                    stagnant_scroll_count = 0

                if stagnant_scroll_count >= max_stagnant_scrolls:
                    logger.warning(
                        "Batas stagnansi scroll tercapai. Kemungkinan telah mencapai akhir timeline."
                    )
                    break

                if len(collected_data) >= max_tweets:
                    break

                # Dynamic Mouse Wheeling dengan Jitter
                # Gerakan scroll bervariasi antara 600 - 950 px vertikal
                scroll_delta_y = random.randint(600, 950)
                page.mouse.wheel(0, scroll_delta_y)

                # Jeda acak 2.0 - 3.5 detik untuk menghindari bot detection
                jitter = random.uniform(2.0, 3.5)
                logger.debug(f"Scroll delta Y: {scroll_delta_y}px, Jitter delay: {jitter:.2f}s")
                time.sleep(jitter)

        except Exception as e:
            logger.error(f"Terjadi kesalahan saat crawling: {e}", exc_info=True)
        finally:
            browser.close()
            logger.info("Browser session ditutup.")

    # 4. Ekspor Data ke CSV menggunakan Pandas
    export_to_csv(collected_data, output_file)


def export_to_csv(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Menyusun data ke dalam Pandas DataFrame dengan tepat 8 kolom wajib
    dan mengekspornya ke format CSV ber-encoding utf-8-sig.
    """
    # 8 Kolom Wajib
    columns = [
        "Sentence",
        "Sentiment",
        "Tweet Date",
        "English Translation",
        "Favorite Count",
        "Retweet Count",
        "Reply Count",
        "Quote Count",
    ]

    if not data:
        logger.warning("Tidak ada data tweet yang berhasil dikumpulkan. Membuat file CSV kosong.")
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(data)
        # Memastikan urutan kolom presisi 8 kolom
        df = df[columns]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Berhasil mengekspor {len(df)} baris data ke '{output_path}' (encoding: utf-8-sig).")


def main():
    # Memuat variabel dari file .env
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Playwright Stealth Crawler untuk X (Twitter) dengan Anti-Virtual DOM Recycling."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL target (jika tidak diisi, otomatis menggunakan TARGET_URL dari .env atau dibuat dari --since dan --until).",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Filter tanggal awal tweet (format YYYY-MM-DD, contoh: 2026-09-28).",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Filter tanggal akhir tweet (format YYYY-MM-DD, contoh: 2026-10-05).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="Daftar ticker saham (default: 10 Big Cap IDX).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["top", "live"],
        default="top",
        help="Mode pencarian X: 'top' untuk tren relevan, 'live' untuk kronologis terbaru.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=int(os.getenv("MAX_TWEETS", "50")),
        help="Jumlah maksimum tweet yang ingin diekstrak.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=os.getenv("HEADLESS", "true").lower() == "true",
        help="Jalankan browser dalam mode headless (default: True).",
    )
    parser.add_argument(
        "--headed",
        dest="headless",
        action="store_false",
        help="Jalankan browser dalam mode GUI/headed (terlihat di layar).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Nama berkas keluaran CSV (default: auto generate di data/raw/ atau dari .env).",
    )

    args = parser.parse_args()

    # Tentukan Target URL
    if args.url:
        target_url = args.url
    elif args.since or args.until or args.tickers:
        target_url = build_search_url(
            tickers=args.tickers,
            since_date=args.since,
            until_date=args.until,
            filter_mode=args.mode,
        )
    else:
        target_url = os.getenv("TARGET_URL", "https://x.com/home")

    # Tentukan Output Path
    if args.output:
        output_file = args.output
    elif args.since and args.until:
        from datetime import datetime
        today_tag = datetime.now().strftime("%Y-%m-%d")
        output_file = f"data/raw/x_stock_tweets_since_{args.since}_until_{args.until}_{today_tag}.csv"
    else:
        output_file = os.getenv("OUTPUT_FILE", "data/raw/tweets_output.csv")

    logger.info(f"Menggunakan target URL: {target_url}")
    logger.info(f"Menyimpan hasil ke: {output_file}")

    run_crawler(
        target_url=target_url,
        max_tweets=args.max,
        headless=args.headless,
        output_file=output_file,
    )



if __name__ == "__main__":
    main()
