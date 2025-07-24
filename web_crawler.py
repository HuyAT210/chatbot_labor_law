import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from core.rag_chain import ask_llm
import hashlib
import time
from config.config import QWEN_API_URL, QWEN_API_KEY, QWEN_MODEL
import datetime

START_URL = "https://www.usa.gov/labor-laws"
DOMAIN = "usa.gov"
OUTPUT_DIR = "crawled_data"

# At the start of the script, create a session folder with a timestamp
SESSION_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(OUTPUT_DIR, SESSION_TIME)
os.makedirs(SESSION_DIR, exist_ok=True)
LOG_FILE = os.path.join(SESSION_DIR, "crawl_log.txt")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

visited = set()
CRAWL_DATA_FILE = os.path.join(SESSION_DIR, "crawl_data.txt")


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")


def url_to_filename(url, ext=".txt"):
    # Use a hash to avoid filesystem issues
    h = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(SESSION_DIR, f"{h}{ext}")


def save_content(url, content, ext=".txt"):
    # Append all content to a single crawl_data.txt file
    with open(CRAWL_DATA_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\nURL: {url}\n{'='*80}\n{content}\n")
    log(f"Appended {url} to {CRAWL_DATA_FILE}")


def should_use_llm(url):
    # If not in usa.gov, use LLM
    return DOMAIN not in urlparse(url).netloc


def extract_links(soup, base_url):
    internal_links = []
    external_links = []
    # Collect all <a href> links
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        full_url = urljoin(base_url, href)
        if DOMAIN in urlparse(full_url).netloc:
            # Only follow <a class="text-no-underline"> within usa.gov
            if "text-no-underline" in (a.get("class") or []):
                internal_links.append(full_url)
        else:
            external_links.append(full_url)
    return internal_links, external_links


def is_labor_law_page_with_llm(html, url):
    import requests
    import json
    # Use a short snippet to avoid overloading the LLM
    snippet = html[:2000]
    prompt = (
        f"Is the following web page about labor law or employment law? Answer YES or NO.\n\nURL: {url}\n\nHTML snippet:\n{snippet}"
    )
    payload = {
        "model": QWEN_MODEL if 'QWEN_MODEL' in globals() else "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip().upper()
        return content.endswith("YES")
    except Exception as e:
        log(f"LLM relevance check error for {url}: {e}")
        return False

def extract_text_with_llm(html, url):
    import requests
    import json
    prompt = (
        "Extract ONLY the main readable text that is DIRECTLY about labor law or employment law from this HTML. "
        "If the page is not about labor law or employment law, respond with ONLY: NOT LABOR LAW.\n\n"
        f"URL: {url}\n\nHTML:\n{html}"
    )
    payload = {
        "model": QWEN_MODEL if 'QWEN_MODEL' in globals() else "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        if not result or result.upper().startswith("NOT LABOR LAW"):
            log(f"LLM extraction for {url} returned NOT LABOR LAW or empty. Skipping save.")
            return None
        return result
    except Exception as e:
        log(f"LLM extraction error for {url}: {e}")
        return None

SOCIAL_DOMAINS = [
    'facebook.com', 'twitter.com', 'x.com', 'linkedin.com', 'instagram.com',
    'youtube.com', 'tiktok.com', 'pinterest.com', 'reddit.com', 'wa.me', 'web.whatsapp.com',
    'plus.google.com', 'tumblr.com', 'snapchat.com', 'weibo.com', 'vk.com', 'line.me',
    'mailto:', 'javascript:'
]

def is_valid_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    for domain in SOCIAL_DOMAINS:
        if domain in url:
            return False
    return True

def crawl(url):
    if url in visited:
        return
    if not is_valid_url(url):
        log(f"Skipping non-http(s) or social/media URL: {url}")
        return
    visited.add(url)
    log(f"Crawling: {url}")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        final_url = resp.url
        is_usa_gov = DOMAIN in urlparse(final_url).netloc
        if not is_usa_gov:
            # LLM relevance check before extraction
            if is_labor_law_page_with_llm(resp.text, final_url):
                log(f"External site {final_url} is labor law related. Using LLM for extraction.")
                llm_result = extract_text_with_llm(resp.text, final_url)
                if llm_result:
                    save_content(final_url, llm_result, ext=".llm.txt")
                    log(f"LLM processed {final_url}")
                else:
                    log(f"LLM extraction for {final_url} was not labor law related. Not saving.")
            else:
                log(f"External site {final_url} is NOT labor law related. Skipping.")
            return  # Do NOT extract or follow any links from external sites
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        save_content(final_url, text)
        # Extract both internal and external links
        internal_links, external_links = extract_links(soup, final_url)
        # Crawl external links (no recursion)
        for link in external_links:
            if link not in visited and is_valid_url(link):
                crawl(link)
        # Recursively crawl internal links
        for link in internal_links:
            if link not in visited and is_valid_url(link):
                crawl(link)
    except Exception as e:
        log(f"Error crawling {url}: {e}")


def main():
    log("Starting crawl...")
    crawl(START_URL)
    log("Crawl finished.")

if __name__ == "__main__":
    main()
