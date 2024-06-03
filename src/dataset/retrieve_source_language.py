import bs4
import csv
import requests
from storage_config import *

UMLS_VOCABULARY_URL = "https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html"

if __name__ == "__main__":
    req = requests.get(UMLS_VOCABULARY_URL)
    soup = bs4.BeautifulSoup(req.text, "html.parser")
    table = soup.find_all('table')[0]
    rows = table.find_all("tr")

    with open(VOCAB_PATH, 'w', newline='', encoding="utf-8") as vocab_csv:
        writer = csv.writer(vocab_csv)
        for i in range(len(rows)):
            cells = rows[i].find_all("th") if i == 0 else rows[i].find_all("td")
            csv_line = [cell.text.strip() for cell in cells]
            writer.writerow(csv_line)
