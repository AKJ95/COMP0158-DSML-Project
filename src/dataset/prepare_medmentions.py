# Import built-in libraries
from enum import Enum
import re

# Import additional libraries
import spacy
from tqdm import tqdm

# Import project code
from storage_config import *


class LineStatus(Enum):
    ANNOTATION = 0
    TITLE = 1
    ABSTRACT = 2


def is_pubid(pubid: str) -> bool:
    match_flag = re.match(pattern=r'^[0-9]{8}$', string=pubid) is not None
    return match_flag


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    with open(MEDMENTIONS_FULL_PATH) as f:
        status = LineStatus.ANNOTATION
        current_pubid = ""
        documents = []
        preprocessed_lines = []
        tokens = []
        sentence_starting_indices = []
        current_token_index = 0
        current_sentence_id = 0
        for line in tqdm(f, desc="Parsing MedMentions data..."):
            # Update status of type of the line being processed
            if is_pubid(line[:8]) and line[:8] != current_pubid:
                status = LineStatus.TITLE
            elif status == LineStatus.TITLE:
                status = LineStatus.ABSTRACT
            elif status == LineStatus.ABSTRACT:
                status = LineStatus.ANNOTATION

            if status == LineStatus.TITLE:
                # Process any remaining tokens from the previous document
                for i in range(current_token_index, len(tokens)):
                    if i in sentence_starting_indices:
                        current_sentence_id += 1
                    token_text = tokens[i].text
                    preprocessed_lines.append([current_pubid, current_sentence_id, token_text, tokens[i].pos_, "O"])
                # Update variables for the new document
                current_pubid = line[:8]
                title = line.split("|")[-1][:-1]
                documents.append(title)
            elif status == LineStatus.ABSTRACT:
                # Retrieve abstract and concatenate with its title.
                num_tokens_in_title = len(list(nlp(documents[-1])))
                abstract = line.split("|")[-1][:-1]
                documents[-1] = " ".join([documents[-1], abstract])
                # Parse the combined document and retrieve tokens.
                parsed_document = nlp(documents[-1])
                tokens = [token for token in parsed_document]
                # Initialise variables before processing annotations.
                current_token_index = 0
                sentence_starting_indices = [sent.start for sent in parsed_document.sents]
                sentence_starting_indices.append(num_tokens_in_title)
            elif is_pubid(line[:8]):
                # Parse annotation line. Retrieve starting and ending indices of the annotated string.x
                line_elements = line.split("\t")
                starting_index = int(line_elements[1])
                ending_index = int(line_elements[2])
                # Process tokens of the current document
                for i in range(current_token_index, len(tokens)):
                    token_text = tokens[i].text
                    token_index = tokens[i].idx
                    # All tokens preceding the annotation should be recorded as "O"
                    if token_index < starting_index:
                        if i in sentence_starting_indices:
                            current_sentence_id += 1
                        preprocessed_lines.append([current_pubid, current_sentence_id, token_text, tokens[i].pos_, "O"])
                    # The token that matches the starting index of the annotation should be recorded as "B"
                    elif token_index == starting_index:
                        if i in sentence_starting_indices:
                            current_sentence_id += 1
                        preprocessed_lines.append([current_pubid, current_sentence_id, token_text, tokens[i].pos_, "B"])
                    # Any tokens after the starting index but in front of the ending index should be recorded as "I"
                    elif token_index < ending_index:
                        if i in sentence_starting_indices:
                            current_sentence_id += 1
                        preprocessed_lines.append([current_pubid, current_sentence_id, token_text, tokens[i].pos_, "I"])
                    # If the token appears after the annotation, stop parsing the tokens until the next line is read.
                    else:
                        current_token_index = i  # Save the current position of processed token.
                        break

        # Parse any remaining tokens after all annotation lines are read.
        for i in range(current_token_index, len(tokens)):
            if i in sentence_starting_indices:
                current_sentence_id += 1
            token_text = tokens[i].text
            preprocessed_lines.append([current_pubid, current_sentence_id, token_text, tokens[i].pos_, "O"])

        # Once parsing is completed, print message and summary statistics.
        print("Completed parsing MedMentions Data.")
        print(f"Number of PubMed articles: {len(documents)}")
        print(f"Number of sentences: {current_sentence_id}")
        print(f"Number of tokens: {len(preprocessed_lines)}")
