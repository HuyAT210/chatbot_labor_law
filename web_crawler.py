#!/usr/bin/env python3
"""
AI-Powered Web Crawler for AI Document Assistant
Uses OpenAI GPT to intelligently navigate websites and process content before database storage
"""

import requests
from bs4 import BeautifulSoup
import time
import os
import re
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path
import argparse
from typing import List, Dict, Set, Tuple, Optional
import logging
import openai
from config.config import OPENAI_API_KEY

# Set up OpenAI
openai.api_key = OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIContentProcessor:
    """AI-powered content filtering only (no summarization or restructuring)"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def is_page_relevant(self, url: str, title: str, content: str, target_domain: str) -> bool:
        """
        Use AI to determine if a page contains substantive immigration law knowledge.
        Returns True only if the content contains actual legal information, procedures, 
        or guidance that would be useful for immigration lawyers helping citizenship applicants.
        """
        try:
            # First, do a quick content check to filter out obvious non-content
            content_lower = content.lower()
            if any(skip in content_lower for skip in [
                'rss feed', 'was this page helpful', 'yes no', 'select a reason',
                'how can the content be improved', 'skip to main content',
                'official government website', 'secure website', 'español',
                'multilingual resources', 'sign in', 'create account',
                'menu', 'close menu', 'breadcrumb', 'return to top',
                'facebook', 'twitter', 'youtube', 'instagram', 'linkedin',
                'email', 'contact uscis', 'privacy policy', 'terms of use',
                'accessibility', 'freedom of information act', 'no fear act',
                'inspector general', 'white house', 'usa.gov', 'vote.gov'
            ]):
                return False
            
            # Check if content is too short or seems like navigation
            if len(content.strip()) < 200:
                return False
            
            prompt = f"""
            You are an expert immigration lawyer. Determine if this webpage contains SUBSTANTIVE legal information that would help you guide a client through the US citizenship process.

            ONLY answer YES if the page contains:
            - Specific legal requirements, procedures, or eligibility criteria
            - Immigration law statutes, regulations, or policy guidance
            - Step-by-step instructions for forms or processes
            - Legal definitions, interpretations, or clarifications
            - Official policy memos, field guidance, or legal precedents
            - Specific eligibility requirements, time periods, or qualifications
            - Legal consequences, penalties, or rights
            - Official procedures, timelines, or processes

            Answer NO if the page contains:
            - General information, news, or announcements
            - Navigation menus, feedback forms, or website features
            - Contact information, office locations, or administrative details
            - General descriptions without specific legal content
            - Links pages, sitemaps, or organizational information
            - Social media, accessibility, or website policy information

            URL: {url}
            Title: {title}
            Content Preview: {content[:1500]}...

            Answer only YES or NO:
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            
            answer = response.choices[0].message.content.strip().lower()
            return answer.startswith("yes")
            
        except Exception as e:
            logger.error(f"❌ AI relevance check failed: {e}")
            return False

class AIWebCrawler:
    def __init__(self, output_dir: str = "crawled_data", delay: float = 2.0, max_pages: int = 30):
        """
        Initialize the AI-powered web crawler
        
        Args:
            output_dir: Directory to save processed content
            delay: Delay between requests in seconds
            max_pages: Maximum number of pages to crawl
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.max_pages = max_pages
        self.ai_processor = AIContentProcessor()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.visited_urls: Set[str] = set()
        self.processed_pages: List[Dict] = []
        self.crawl_queue: List[Tuple[str, float]] = []  # (url, priority)
        
    def extract_page_content(self, url: str) -> Dict[str, str]:
        """Extract raw content from a webpage with better organization"""
        try:
            logger.info(f"📄 Crawling: {url}")
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove more non-content elements
            for element in soup([
                "script", "style", "nav", "footer", "header", "aside",
                "form", "button", "input", "select", "textarea",
                ".breadcrumb", ".navigation", ".menu", ".sidebar",
                ".social-media", ".feedback", ".help", ".contact",
                ".accessibility", ".privacy", ".terms", ".sitemap"
            ]):
                element.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract content from main areas with better targeting
            content_selectors = [
                'main', 'article', '[role="main"]', '.content', '.main-content',
                '#content', '#main', '.usa-main-content', '.usa-content',
                '.usa-layout-docs__main', '.usa-layout-docs__content',
                '.usa-section', '.usa-grid', '.usa-width-one-whole',
                '.usa-width-two-thirds', '.usa-width-one-half'
            ]
            
            content_parts = []
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 100:  # Only substantial content
                        content_parts.append(text)
            
            # If no main content found, try paragraphs and headings
            if not content_parts:
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 50:  # Only meaningful content
                        content_parts.append(text)
            
            # Fallback to body content
            if not content_parts:
                body = soup.find('body')
                if body:
                    content_parts.append(body.get_text())
            
            raw_content = '\n\n'.join(content_parts)
            
            # Clean up the content
            raw_content = self.clean_content(raw_content)
            
            return {
                'title': title,
                'content': raw_content,
                'url': url
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting content from {url}: {e}")
            return {'title': '', 'content': '', 'url': url}
    
    def clean_content(self, content: str) -> str:
        """Clean and organize the extracted content"""
        if not content:
            return ""
        
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove common navigation and non-content text
        content = re.sub(r'(Skip to main content|U\.S\. flag|Official Government Website|Secure Website|Español|Multilingual Resources|Sign In|Create Account|Menu|Close menu|Breadcrumb|Return to top|Facebook|X, formerly known as Twitter|YouTube|Instagram|LinkedIn|Email|Contact USCIS|U\.S\. Department of Homeland Security Seal|Agency description|Important links|Looking for U\.S\. government information and services\?|Visit USA\.gov|Was this page helpful\?|Yes|No|This page was not helpful because the content|Select a reason|has too little information|has too much information|is confusing|is out of date|other|How can the content be improved\?|To protect your privacy, please do not include any personal information in your feedback|Review our Privacy Policy|RSS Feed|Subscribe|Follow us|Share this page|Print this page|Download|PDF|Word|Excel|PowerPoint|Accessibility|Privacy Policy|Terms of Use|Freedom of Information Act|No Fear Act|Inspector General|White House|USA\.gov|Vote\.gov)', '', content, flags=re.IGNORECASE)
        
        # Remove very short lines that are likely navigation
        lines = [line.strip() for line in content.split('\n') if len(line.strip()) > 20]
        
        # Remove lines that are just punctuation, numbers, or navigation
        lines = [line for line in lines if not re.match(r'^[\d\s\-\.]+$', line)]
        lines = [line for line in lines if not re.match(r'^(Home|About|Contact|Help|Search|Menu|Close|Back|Next|Previous)$', line, re.IGNORECASE)]
        
        return '\n'.join(lines)
    
    def extract_links(self, url: str, base_domain: str) -> List[str]:
        """Extract links from a page"""
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                if urlparse(full_url).netloc == base_domain:
                    if not any(skip in full_url.lower() for skip in [
                        'javascript:', 'mailto:', 'tel:', '#', '/search', '/contact', '/feedback'
                    ]):
                        links.append(full_url)
            
            return list(set(links))
            
        except Exception as e:
            logger.error(f"❌ Error extracting links from {url}: {e}")
            return []
    
    def intelligent_crawl(self, start_url: str, site_name: str) -> List[Dict]:
        """
        Perform AI-powered intelligent crawling, but only filter for relevance (no summarization)
        """
        logger.info(f"🧠 Starting AI-powered intelligent crawl of {site_name} (filter only, no summarization)")
        
        base_domain = urlparse(start_url).netloc
        self.crawl_queue = [(start_url, 10.0)]  # Start with high priority
        
        while self.crawl_queue and len(self.processed_pages) < self.max_pages:
            self.crawl_queue.sort(key=lambda x: x[1], reverse=True)
            current_url, priority = self.crawl_queue.pop(0)
            if current_url in self.visited_urls:
                continue
            self.visited_urls.add(current_url)
            # Extract raw content
            page_data = self.extract_page_content(current_url)
            if not page_data['content'] or len(page_data['content']) < 100:
                continue
            # AI relevance filter only
            logger.info(f"🤖 AI filtering: {page_data['title'][:50]}...")
            is_relevant = self.ai_processor.is_page_relevant(
                current_url, page_data['title'], page_data['content'], base_domain
            )
            if is_relevant:
                self.processed_pages.append({
                    'title': page_data['title'],
                    'url': page_data['url'],
                    'content': page_data['content']
                })
                logger.info(f"✅ Kept relevant page {len(self.processed_pages)}/{self.max_pages}: {page_data['title'][:50]}...")
                # Get and prioritize links for further crawling
                if len(self.processed_pages) < self.max_pages:
                    links = self.extract_links(current_url, base_domain)
                    if links:
                        # Use AI to prioritize links (optional, or just add all)
                        for link in links:
                            if link not in self.visited_urls:
                                self.crawl_queue.append((link, 5.0))
            # Rate limiting
            time.sleep(self.delay)
        logger.info(f"🎯 AI-powered crawl completed: {len(self.processed_pages)} relevant pages processed")
        return self.processed_pages
    
    def save_processed_content(self, pages: List[Dict], site_name: str) -> str:
        """Save only the full, original content of relevant pages"""
        if not pages:
            logger.warning(f"⚠️ No relevant content to save for {site_name}")
            return ""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        txt_filename = f"{site_name}_relevant_fulltext_{timestamp}.txt"
        txt_filepath = self.output_dir / txt_filename
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Full Original Content of Relevant Pages\n")
            f.write(f"# Source: {site_name}\n")
            f.write(f"# Processed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total relevant pages: {len(pages)}\n\n")
            for i, page in enumerate(pages, 1):
                f.write(f"## Page {i}: {page['title']}\n")
                f.write(f"URL: {page['url']}\n")
                f.write("-" * 80 + "\n")
                f.write(page['content'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        logger.info(f"💾 Saved {len(pages)} relevant pages to {txt_filepath}")
        return str(txt_filepath)

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Web Crawler for Immigration Law Content")
    parser.add_argument("--output", "-o", default="crawled_data", help="Output directory")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="Delay between requests (seconds)")
    parser.add_argument("--max-pages", "-m", type=int, default=30, help="Maximum pages to crawl")
    
    args = parser.parse_args()
    
    # Initialize AI crawler
    crawler = AIWebCrawler(
        output_dir=args.output,
        delay=args.delay,
        max_pages=args.max_pages
    )
    
    # USCIS Laws and Policy URL
    uscis_url = "https://www.uscis.gov/laws-and-policy"
    site_name = "uscis_laws_policy"
    
    try:
        logger.info(f"🚀 Starting AI-powered crawl of USCIS Laws and Policy")
        logger.info(f"🧠 Using AI for intelligent navigation and content processing")
        logger.info(f"📊 Target: Up to {args.max_pages} pages with AI analysis")
        
        pages = crawler.intelligent_crawl(uscis_url, site_name)
        
        if pages:
            filepath = crawler.save_processed_content(pages, site_name)
            total_chars = sum(len(page['content']) for page in pages)
            
            logger.info(f"✅ AI-powered crawl completed successfully!")
            logger.info(f"📈 Results: {len(pages)} pages, {total_chars:,} characters")
            logger.info(f"💾 RAG-ready content saved to: {filepath}")
        else:
            logger.warning(f"⚠️ No content found for USCIS")
            
    except Exception as e:
        logger.error(f"❌ Error in AI-powered crawl: {e}")
    
    logger.info("🎉 AI-powered crawling completed!")

if __name__ == "__main__":
    main() 