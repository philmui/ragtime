###############################################################################
# crawl_sites.py
# --------------
#
# Web crawler for scraping content from a set of URLs
# (does not crawl beyond TLD)
#
# Usage:
#     python crawl_sites.py -d <depth> url1, url2, ...
#
# For help:
#     python crawl_sites.py -h
#
# Example usage:
# > python -m crawler.crawl_sites  -d 1 -r request_with_retry \
#     -u rss_feed https://www.newswire.com/newsroom/rss/medicine-and-healthcare
# > python -m crawler.crawl_sites  -d 1 -r request_with_retry \
#    -u spider https://www.fda.gov/news-events/fda-newsroom/press-announcements
#
# Phil Mui <pmui@salesforce.com>
#
# Sat Jan 27 10:29:25 PST 2024
###############################################################################

import aiohttp
import argparse
import asyncio
from bs4 import BeautifulSoup
from collections import deque
import hashlib
import json
import os
import re
import random
import time
from urllib.parse import urljoin, urlparse
from .request_html import requestWithRetry, HTTP_RETRIES
from enum import Enum
from typing import List, Tuple
import asyncio
from playwright.async_api import async_playwright

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crawler")

from .markdown import MarkdownConverterFactory

from scrapingant_client import ScrapingAntClient
SCRAPINGANT_API_TOKEN = os.environ.get("SCRAPINGANT_API_TOKEN")
ant_client = ScrapingAntClient(token=SCRAPINGANT_API_TOKEN)

RequestType = Enum("RequestType", ["proxy", "request_with_retry"])
UrlType = Enum("UrlType", ["spider", "rss_feed"])

_OUTPUT_FILE:str = "output.md"
_DEFAULT_DEPTH:int = 1
_NO_PARENTS:bool = False

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


_FILTER_TITLE = {
    'About',
    'Contact Us',
    'Terms of Service'
}

_FILTER_TITLE = {s.lower() for s in _FILTER_TITLE }

_FILTER_URL = {
    'youtube.com',
    'www.youtube.com',
    'linkedin.com',
    'www.linkedin.com',
    'facebook.com',
    'www.facebook.com',
    'instagram.com',
    'www.instagram.com'
}

_FILTER_URL_PATH = {
    'contact-us',
    'sitemap',
    'login'
}
_URL_FULLY_PROCESSED = 2
_URL_IN_QUEUE = 1
    

###############################################################################
# utility functions
###############################################################################

def get_file_id_hash(
    str_description: str
    ) -> str:
    
    assert(str_description is not None), "file id is None"
    m = hashlib.md5()
    m.update(str_description.encode('utf-8'))
    return str(int(m.hexdigest(), 16))[0:12]

def get_file_id(
    url: str
    ) -> str:

    # get the file id using full url
    assert(url is not None), "file id's url is None"
    url_path = urlparse(url)
    path = url_path.path
    path = path.strip('/').replace('/', '_')
    domain = url_path.netloc
    domain = domain.replace('.', '_')
    return f'{domain}_{path}'

def is_filtered_url(
    url: str, 
    start_url: str
    ) -> bool:
    
    assert(url is not None), "is_filtered_url received None url"
    assert(start_url is not None), "is_filtered_url received None start_url"

    if 'attachment' in url:
        return True

    url_parts = urlparse(url)
    domain = url_parts.netloc
    if domain in _FILTER_URL:
        return True

    # filter all urls which are not children of the seed url
    start_url_parts = urlparse(start_url)
    if start_url_parts.netloc != domain:
        return True

    url_clean = url.strip('/').lower()
    for filter_path in _FILTER_URL_PATH:
        if url_clean.endswith(filter_path):
            return True

    return False

def save_visited(
    visited: dict,
    VISITED_FILE: str
    ) -> bool:
    
    assert(visited is not None), "save_visited received None visited"
    assert(VISITED_FILE is not None), "save_visited received None VISITED_FILE"

    try:
        with open(VISITED_FILE, 'w') as file:
            json.dump(visited, file)
            file.write('\n')
    except Exception as e:
        logger.error(f"save_visited: {e}")
        return False

def load_visited(
    VISITED_FILE: str
    ) -> dict:
    
    assert(VISITED_FILE is not None), "load_visited received None VISITED_FILE"
    try:
        with open(VISITED_FILE, 'r') as file:
            data_loaded = json.load(file)
            return data_loaded
    except Exception as e:
        # not an error -- just that this is a new file
        logger.info(f"creating a new visited file: {VISITED_FILE}")
        return {}
    
###############################################################################
# html fetching
###############################################################################

class ContentType(Enum):
    TEXT = 'text'
    MARKDOWN = 'markdown'

def get_content(
    soup : BeautifulSoup, 
    url: str,
    content_type = ContentType.MARKDOWN
    ) -> dict:
    
    assert(soup is not None), "get_content received None soup"

    title = ''
    if soup.title:
        title = soup.title.string
        title = _RE_COMBINE_WHITESPACE.sub(" ", title).strip()

    if content_type == ContentType.MARKDOWN:
        html_content = soup.prettify()
        content = MarkdownConverterFactory.get_converter().convert(html_content)
    else:
        text = soup.get_text(separator=" ", strip=True)
        content = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        content = _RE_COMBINE_WHITESPACE.sub(" ", content).strip()

    # if content and len(content.split()) > 100 and title and not is_filtered_title(title):
    return {
        "url": url,
        "title": title,
        "content": content
    }


async def get_html_from_url(
    url: str,
    request_type: RequestType
    ) -> dict:
    
    assert(url is not None), "get_html_from_url received None url"
    time.sleep(random.uniform(0.05, 0.1))
    
    html = ""
    try:
        if request_type == RequestType.proxy:
            result = await ant_client.general_request_async(url)
            html = result.content
        elif request_type == RequestType.request_with_retry:
            html = requestWithRetry(url, HTTP_RETRIES)
        else:
            raise Exception('Request type is invalid.')
    except Exception as e:
        print(f"Failed get_html_from_url: {e} -> {url}")

    return html


async def playwright_fetch(url: str):
    html = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(java_script_enabled=True, is_mobile=True)
        await page.goto(url, wait_until='domcontentloaded')
        await page.wait_for_timeout(1000)
        html = await page.content()
        await browser.close()
    return html


async def smart_html_from_url(
    url: str,
    ) -> Tuple[str, str]:

    """Fetch HTML content from a URL with error handling and retries.

    :param url: The URL to fetch HTML content from.
    :type url: str
    :return: (The fetched HTML content, Markdown of that content)
    :rtype: Tuple[str, str]
    """
    
    assert(url is not None), "get_html_from_url received None url"
    
    page_html = page_md = ""
    try:
        page_html = requestWithRetry(url, HTTP_RETRIES)
        page_md = MarkdownConverterFactory.get_converter().convert(
            html=page_html
        ).strip()
        page_md = re.sub(r'\n+', '\n', page_md)
    except Exception as e:
        logger.info(f"\t==> failed simple request for {url}")

    # comment: playwright does not retrieve some of the pages yet
    # if len(page_md) < 100:
    #     logger.info(f"\t==> using playwright for {url}")
    #     page_html = await playwright_fetch(url)
    #     page_md = MarkdownConverterFactory.get_converter().convert(
    #         html=page_html
    #     ).strip()
    #     page_md = re.sub(r'\n+', '\n', page_md)

    if len(page_md) < 100:
        logger.info(f"\t==> using ant for {url}")
        result = await ant_client.general_request_async(url)
        page_html = result.content
        page_md = MarkdownConverterFactory.get_converter().convert(
            html=page_html
        ).strip()
        page_md = re.sub(r'\n+', '\n', page_md)

    return page_html, page_md


###############################################################################
# links fetching
###############################################################################
    
async def get_links(
    parent_url: str,  # The parent URL to fetch links from
    start_url: str,  # The initial URL to start the crawling process
    url_level: int,  # The max link level to crawl
    visited_url: dict,  # A dictionary to keep track of visited URLs
    max_links: int  # The maximum number of links to fetch
) -> Tuple[str, set]:  # Returns a tuple containing the page metadata and a set of links
    """
    Fetch links from a given URL and return the links and the page content.
    The links are added to the visited_url dictionary and the page content is returned as a string.

    Args:
        parent_url (str):  The parent URL to fetch links from
        start_url (str): The initial URL to start the crawling process
        url_level (int): The max link level to crawl
        visited_url (dict): A dictionary to keep track of visited URLs
        max_links (int): The maximum number of links to fetch
    Returns:
        Tuple[str, set]: a tuple of (page content str, a set of links on page)
    """

    # Logging the function call with the parent URL
    logger.info(f"\tget_links: {parent_url} ...")

    # Ensuring the function parameters are valid
    assert(parent_url is not None), "get_links received None parent_url"
    assert(start_url is not None), "get_links received None start_url"
    assert(url_level >= 0), "get_links received url_level <= 0"

    links = set()
    page_md = ""
    try:
        page_html, page_md = await smart_html_from_url(parent_url)
        
        if page_html is None or len(page_html.strip()) < 150:
            return links

        page_html = re.sub(r'\n+|\s+', lambda m: ' ' if m.group() else '', page_html)
        logger.info(url_level * "." + f'{url_level}: {parent_url}: {len(page_html)}')

        soup = BeautifulSoup(page_html, 'html.parser')
        for link in soup.find_all('a')[:max_links]:
            path = link.get('href')

            if path is None or not (path.startswith('http') or path.startswith('/')):
                continue
            if path and path.startswith('/'):
                path = urljoin(parent_url, path)

            if path not in visited_url and not is_filtered_url(path, start_url):
                links.add(path)

        # set the status of url as fully processed
        visited_url[parent_url] = _URL_FULLY_PROCESSED

    except BaseException as e:
        logger.error(f"Failed get_links: {e}")

    logger.info(f"\t... done get_links: {parent_url[-20:]}")

    return page_md, links


###############################################################################
# crawl
###############################################################################

async def crawl(
    start_url: str, 
    level: int,
    visited: dict,
    parent_url: str = None,
    max_pages: int = 20
    ) -> Tuple[dict, List[str]]:
    
    assert(start_url is not None), "crawl received None start_url"
    assert(level >= 0), "get_links received level < 0"
    assert(visited is not None), "crawl received None visited"

    q = deque()
    q.append((start_url, 0))
    page_content_list = []
    while q and len(page_content_list) < max_pages:
        logger.info(f"crawl queue: {len(q)}")

        async with aiohttp.ClientSession() as session:
            # all links at the queue should be at the same level right now: NEED TO CHECK
            current_level = q[0][1]

            # there should not be any race condition for update to
            # visited as we process unique urls at every level
            page_links_array = await asyncio.gather(
                *[get_links(url, start_url, url_level, visited, max_pages) for url, url_level in q]
            )
            q.clear()
            # first save all page_content
            for page_content, links in page_links_array:
                page_content_list.append(page_content)

            if current_level < level:
                # then figure out whether to add the links to the queue
                for page_content, links in page_links_array:
                    if links is not None and len(links) > 0:
                        for link in links:
                            parsed_link = urlparse(link)._replace(fragment='') # remove fragment
                            parsed_link_url = parsed_link.geturl()
                            if parsed_link_url not in visited.keys() and \
                               (parent_url is None or parsed_link_url.startswith(parent_url)):
                                # set the status of url as ready to process
                                q.append((parsed_link_url, current_level + 1))
                                visited[parsed_link_url] = _URL_IN_QUEUE

    return visited, page_content_list


def setup_crawl(
    start_url: str, 
    depth: int,
    url_type: UrlType
    ) -> None:
    """Start processing URLs

    Args:
        start_url (str): _description_
        depth (int): _description_
    """
    assert(start_url is not None), "process received None start_url"
    assert(depth >= 0 and depth < 10), "process received depth < 0 or depth > 9"

    global _OUTPUT_FILE
    _OUTPUT_FILE = get_file_id(start_url) + '.out'
    if os.path.exists(_OUTPUT_FILE):
        mode = 'a'
    else:
        mode = 'w'
    with open(_OUTPUT_FILE, mode) as file:
        file.write(f"")

    VISITED_FILE = _OUTPUT_FILE + ".visited"
    visited = load_visited(VISITED_FILE)

    if url_type == UrlType.spider:
        parsed_url = urlparse(start_url)
        parent_url = parsed_url.scheme + '://' + parsed_url.netloc +\
                        "/".join(parsed_url.path.split("/")[:-1])

        visited, page_content_list = asyncio.run(
            crawl(start_url, depth, visited, parent_url=parent_url, max_pages=20)
        )

    if len(visited) > 0:
        save_visited(visited,VISITED_FILE=VISITED_FILE)

    
###############################################################################
# main()
###############################################################################

if __name__ == "__main__":
    start = time.time()
    _argparser = argparse.ArgumentParser(description='crawl a set of URLs')
    _argparser.add_argument('urls', metavar='url', type=str, nargs='+',
                        help='an URL to crawl')
    _argparser.add_argument('-d', metavar='depth', type=int, default=_DEFAULT_DEPTH,
                        help=f"depth level for processing the URLs (default={_DEFAULT_DEPTH})")
    _argparser.add_argument('--no-parents', action='store_true',
                        help=f"if set no parents means no parent urls are processed (default=False)")
    _argparser.add_argument('-u', metavar='url_type',
                            type=lambda arg: UrlType[arg], choices=UrlType,
                            default="spider",
                            help=f"url type (default=spider)")
    args = _argparser.parse_args()

    if args.no_parents: _NO_PARENTS = True
    print(f"Depth: {args.d}, request_type: {args.r}, url_type: {args.u}")
    for url in args.urls:
        print(f"{20*'#'}site: {url}{20*'#'}")
        setup_crawl(url, args.d, args.u)

    end = time.time()
    print(f'time elapsed: {(end - start)/60.} minutes')

