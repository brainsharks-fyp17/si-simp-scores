import os
import re
from pathlib import Path
from urllib import request

import wget
from bs4 import BeautifulSoup

urls = ["http://citizenslanka.org/law-category-si/family-law-si/?lang=si"]
save_dir = "./scraped"
url_list = []
for url in urls:
    response = request.urlopen(url).read()
    soup = BeautifulSoup(response, "html.parser")
    links = soup.find_all('a', href=re.compile(r'(.pdf)'))  # scrape pdf

    # clean the pdf link names

    for el in links:
        if el['href'].startswith('http'):
            url_list.append(el['href'])

url_list = list(set(url_list))  # remove duplicates
print("Links found:", len(url_list))
Path(save_dir).mkdir(parents=True, exist_ok=True)

# download the pdfs to a specified location
for url_pdf in url_list:
    save_file_name = os.path.join(save_dir, url_pdf.split('/')[-1])
    wget.download(url_pdf, out=save_file_name)
