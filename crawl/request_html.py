#########################################################################
#
# request_html.py
#
# @author: Phil Mui
# @email: thephilmui@gmail.com
# @date: Mon Jan 23 16:45:56 PST 2023
#
# Retry logic: bit.ly/requests-retry
#########################################################################

import os
import random
import time

import aiohttp
import asyncio
from typing import Tuple
import requests
from requests.adapters import HTTPAdapter, Retry

BAD_HTTP_CODES = (400,401,403,404,406,408,409,410,429,500,502,503,504)
HTTP_RETRIES = 6

HEADERS  = [
{ 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' },
{ 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' },
{ 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' },
{ 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' },
{ 'User-Agent': 'Mozilla/5.0 (Android 4.4; Mobile; rv:41.0) Gecko/41.0 Firefox/41.0' },
{ 'User-Agent': 'Mozilla/5.0 (Android 4.4; Tablet; rv:41.0) Gecko/41.0 Firefox/41.0' },
{ 'User-Agent': 'Mozilla/5.0 (X11; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0' },
{ 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0' },
{ 'User-Agent': 'Mozilla/5.0 (Maemo; Linux armv7l; rv:10.0) Gecko/20100101 Firefox/10.0 Fennec/10.0' },
{ 'User-Agent': 'Mozilla/5.0 (Linux; Android 7.0) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Focus/1.0 Chrome/59.0.3029.83 Mobile Safari/537.36' },
{ 'User-Agent': 'Mozilla/5.0 (Linux; Android 7.0) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Focus/1.0 Chrome/59.0.3029.83 Safari/537.36' },
{ 'User-Agent': 'Mozilla/5.0 (Android 7.0; Mobile; rv:62.0) Gecko/62.0 Firefox/62.0' },
]


async def asyncGet(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # Get the response status code
            status = response.status
            # Read the response body
            body = await response.text().strip()
            print(f"\tasy: {len(body)}: {url}")
            # print(f"\tpre: {status}: {body[:50]} ...")

HTTP_RETRIES = 5

def requestWithRetry(
        url: str, 
        numRetries=HTTP_RETRIES, 
        verbose=True
    ) -> str:
    """
    This function performs an HTTP request with retries for failed requests.
    It takes three arguments: url (str), numRetries (int, default is HTTP_RETRIES),
    and verbose (bool, default is True). It returns the HTML content as a string.

    :param url: The URL to make the HTTP request to
    :type url: str
    :param numRetries: The number of retries for failed requests, defaults to HTTP_RETRIES
    :type numRetries: int, optional
    :param verbose: A flag to enable or disable verbose output, defaults to True
    :type verbose: bool, optional
    :return: The HTML content of the response as a string
    :rtype: str
    """

    html_content = ""
    s = requests.Session()
    retries = Retry(total=2*numRetries,
                    connect=numRetries,
                    read=numRetries,
                    backoff_factor=0.1,
                    status_forcelist=BAD_HTTP_CODES)
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    html = ""
    try:
        response = s.get(url, headers=random.choice(HEADERS))
        response.raise_for_status()
        html_content = response.text.strip()
    except Exception as e:
        print(f"\trequest failed for {url}: {e}")
    s.close()
    if verbose:
        print(f"\treq: {len(html_content)}: {url}: ...{html_content[-7:]}")

    return html_content


