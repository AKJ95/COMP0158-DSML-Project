from collections import defaultdict
import csv
import time
from tqdm import tqdm
from storage_config import *


def get_sources(language_filter: list) -> list:
    sources = []
    with open(VOCAB_PATH, newline='') as vocab_csv:
        reader = csv.reader(vocab_csv, delimiter=',')
        for row in reader:
            if not language_filter or row[3] in language_filter:
                sources.append(row[0])
    return sources


def get_definitions(source_filter: list) -> list:
    definitions = []
    # columns_def = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
    for line in tqdm(open(MRDEF_PATH), desc='Parsing UMLS definitions (MRDEF.RRF)'):
        line = line.rstrip().split('|')
        if not source_filter or line[4] in source_filter:
            definitions.append(line)
    return definitions


if __name__ == "__main__":
    cui = set()
    sui = set()
    umls_strings = set()
    cui_all = set()
    columns_conso = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                     "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
    # columns_def = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
    # for line in tqdm(open(MRDEF_PATH), desc='Parsing UMLS definitions (MRDEF.RRF)'):
    #     line = line.rstrip().split('|')
    #     cui.add(line[0])
    # print(len(cui))
    max_string_length = 0
    for line in tqdm(open(MRCONSO_PATH), desc='Parsing UMLS concepts (MRCONSO.RRF)'):
        line = line.rstrip().split('|')
        if line[1] == "ENG":
            sui.add(line[5])
            umls_strings.add(line[14])
            max_string_length = max(max_string_length, len(line[14]))
    print(f"Number of unique English strings in UMLS: {len(sui)}")
    print(f"Length of ongest string: {max_string_length}")

    english_sources = get_sources(["ENG"])
    definitions = get_definitions(english_sources)
    concepts_with_definition = set([definition[0] for definition in definitions])
    print(f"Number of concepts with English definition: {len(concepts_with_definition)}")

    start = time.time()
    toy_dict = defaultdict(list)
    dict_by_length = []
    for _ in range(max_string_length):
        dict_by_length.append(defaultdict(list))

    count = 0
    for s in umls_strings:
        s_padded = s.center(len(s) + 4, '#')
        s_length = len(s)
        for i in range(len(s_padded)-3):
            dict_by_length[s_length-1][f"{s_padded[i:i+3]}"].append(s_padded)
        # Report Progress
        count += 1
        if count == len(umls_strings) or count % 50000 == 0:
            print(f"Progress: {count}/{len(umls_strings)}; {(count/len(umls_strings)*100):.2f}%")
    end = time.time()
    print(end - start)

    total_count = 0
    count_features = 0
    for toy_dict in dict_by_length:
        count_features += len(toy_dict.keys())
        # for feature in toy_dict.keys():
        #     total_count += len(toy_dict[feature])
    print(f"Average features in one dictionary: {count_features/len(dict_by_length):.2f}")
    # print(f"Average length in each feature: {total_count / count_features:.2f}")
