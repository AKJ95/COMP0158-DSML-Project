# Import built-in libraries
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import math
import time

# Import pip libraries
from tqdm import tqdm

# Import project files
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


def binary_search(sorted_list: list, target: str) -> bool:
    index = bisect_left(sorted_list, target)
    if index != len(sorted_list) and sorted_list[index] == target:
        return True
    else:
        return False


def get_insertion_index_by_binary_search(sorted_list: list, target: str) -> int:
    index = bisect_left(sorted_list, target)
    return index


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
    for _ in range(max_string_length + 3):
        dict_by_length.append(defaultdict(list))

    count = 0
    for s in umls_strings:
        s_padded = s.center(len(s) + 4, '#')
        s_length = len(s)
        for i in range(len(s_padded)-3):
            insertion_index = get_insertion_index_by_binary_search(dict_by_length[s_length+2][f"{s_padded[i:i+3]}"], s)
            current_dict = dict_by_length[s_length+2][f"{s_padded[i:i+3]}"]
            if insertion_index != len(current_dict) \
                    and current_dict != s \
                    or insertion_index == len(current_dict):
                dict_by_length[s_length + 2][f"{s_padded[i:i + 3]}"].insert(insertion_index, s)
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
        for feature in toy_dict.keys():
            total_count += len(toy_dict[feature])
    print(f"Average features in one dictionary: {count_features/len(dict_by_length):.2f}")
    print(f"Average length in each feature: {total_count / count_features:.2f}")
    print()

    # Toy QuickUMLS
    print("Toy QuickUMLS")
    # Pseudomonas aeruginosa (Pa) infection
    start = time.time()
    toy_string = "Pseudomonas aeruginosa (Pa) infection"
    threshold = 0.7
    toy_string_padded = toy_string.center(len(toy_string) + 4, '#')
    min_length = math.ceil((len(toy_string)+2) * threshold)
    max_length = math.floor((len(toy_string)+2) / threshold)
    matched_flag = False
    features = [toy_string_padded[i:i + 3] for i in range(len(toy_string_padded) - 3)]

    for length in range(min_length, max_length+1):
        common_strings = []
        rho = threshold * (length + len(toy_string) + 2) / (1 + threshold)
        m = defaultdict(int)
        features_sorted = sorted(features, key=(lambda x: len(dict_by_length[length][x])))
        for k in range(len(features_sorted) - round(rho) + 1):
            for s in dict_by_length[length][features_sorted[k]]:
                m[s] += 1
        for k in range(len(features_sorted) - round(rho) + 1, len(features_sorted)):
            for s in m.keys():
                if binary_search(dict_by_length[length][features_sorted[k]], s):
                    m[s] += 1
                if m[s] >= rho:
                    matched_flag = True
                    break
            if matched_flag:
                break
        if matched_flag:
            break

    # for length in range(min_length, max_length+1):
    #     common_strings = []
    #     rho = threshold * (length + len(toy_string) + 2) / (1 + threshold)
    #     for i in range(len(toy_string_padded) - 3):
    #         if toy_string_padded[i:i+3] in dict_by_length[length].keys():
    #             common_strings.extend(dict_by_length[length][toy_string_padded[i:i+3]])
    #     string_counter = Counter(common_strings)
    #     most_occurrences = string_counter.most_common(1)[0][1]
    #     if most_occurrences >= rho:
    #         matched_flag = True
    #         break

    end = time.time()
    duration = end - start
    print(f"Elapsed time: {duration:.3f} seconds")
    print(f"Match result: {matched_flag}")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_length}")
