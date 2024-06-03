import nltk
from six.moves import xrange
import spacy
import time

valid_punct = {
    u'\u002d', u'\u007e', u'\u00ad', u'\u058a', u'\u05be', u'\u1400',
    u'\u1806', u'\u2010', u'\u2011', u'\u2010', u'\u2012', u'\u2013',
    u'\u2014', u'\u2015', u'\u2053', u'\u207b', u'\u2212', u'\u208b',
    u'\u2212', u'\u2212', u'\u2e17', u'\u2e3a', u'\u2e3b', u'\u301c',
    u'\u3030', u'\u30a0', u'\ufe31', u'\ufe32', u'\ufe58', u'\ufe63',
    u'\uff0d'
}

negations = {'none', 'non', 'neither', 'nor', 'no', 'not'}

_stopwords = nltk.corpus.stopwords.words()

window = 5

min_match_length = 3


def _is_valid_token(tok):
    return not (
        tok.is_punct or tok.is_space or
        tok.pos_ == 'ADP' or tok.pos_ == 'DET' or tok.pos_ == 'CONJ'
    )


def _is_valid_middle_token(tok):
    return (
            not (tok.is_punct or tok.is_space) or
            tok.is_bracket or
            tok.text in valid_punct
    )


def _is_valid_start_token(tok):
    return not (
            tok.like_num or
            (_is_stop_term(tok) and tok.lemma_ not in negations) or
            tok.pos_ == 'ADP' or tok.pos_ == 'DET' or tok.pos_ == 'CONJ'
    )


def _is_valid_end_token(tok):
    return not (
            tok.is_punct or tok.is_space or _is_stop_term(tok) or
            tok.pos_ == 'ADP' or tok.pos_ == 'DET' or tok.pos_ == 'CONJ'
    )


def _is_stop_term(tok):
    return tok in _stopwords


def _is_longer_than_min(span):
    return (span.end_char - span.start_char) >= min_match_length


def _make_ngrams(sent):
    sent_length = len(sent)

    # do not include determiners inside a span
    skip_in_span = {token.i for token in sent if token.pos_ == 'DET'}

    # invalidate a span if it includes any on these symbols
    invalid_mid_tokens = {
        token.i for token in sent if not _is_valid_middle_token(token)
    }

    for i in xrange(sent_length):
        tok = sent[i] # Current token

        if not _is_valid_token(tok):
            continue

        # do not consider this token by itself if it is
        # a number or a stopword.
        if _is_valid_start_token(tok):
            compensate = False
        else:
            compensate = True

        span_end = min(sent_length, i + window) + 1

        # we take a shortcut if the token is the last one
        # in the sentence
        if (
                i + 1 == sent_length and  # it's the last token
                _is_valid_end_token(tok) and  # it's a valid end token
                len(tok) >= min_match_length  # it's of miminum length
        ):
            yield (tok.idx, tok.idx + len(tok), tok.text)

        for j in xrange(i, span_end): # MODIFIED
            if compensate:
                compensate = False
                continue

            if sent[j - 1] in invalid_mid_tokens:
                break

            if not _is_valid_end_token(sent[j - 1]):
                continue

            span = sent[i:j]

            if not _is_longer_than_min(span):
                continue

            yield (
                span.start_char, span.end_char,
                ''.join(token.text_with_ws for token in span
                        if token.i not in skip_in_span).strip()
            )


def match(text, best_match=True, ignore_syntax=False):
    """Perform UMLS concept resolution for the given string.

    [extended_summary]

    Args:
        text (str): Text on which to run the algorithm

        best_match (bool, optional): Whether to return only the top match or all overlapping candidates. Defaults to True.
        ignore_syntax (bool, optional): Wether to use the heuristcs introduced in the paper (Soldaini and Goharian, 2016).

    Returns:
        List: List of all matches in the text
    """
    parsed = nlp(u'{}'.format(text))

    # pass in parsed spacy doc to get concept matches
    matches = _match(parsed)

    return matches


def _match(doc, best_match=True, ignore_syntax=False):
    """Gathers ngram matches given a spaCy document object.

    [extended_summary]

    Args:
        doc (Document): spaCy Document object to be used for extracting ngrams

        best_match (bool, optional): Whether to return only the top match or all overlapping candidates. Defaults to True.
        ignore_syntax (bool, optional): Wether to use the heuristcs introduced in the paper (Soldaini and Goharian, 2016).

    Returns:
        List: List of all matches in the text
    """

    ngrams = _make_ngrams(doc)

    return ngrams


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    start = time.time()
    sentence = "Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) patients is associated with worse long-term pulmonary disease and shorter survival, and chronic Pa infection (CPA) is associated with reduced lung function, faster rate of lung decline, increased rates of exacerbations and shorter survival. By using exome sequencing and extreme phenotype design, it was recently shown that isoforms of dynactin 4 (DCTN4) may influence Pa infection in CF, leading to worse respiratory disease. The purpose of this study was to investigate the role of DCTN4 missense variants on Pa infection incidence, age at first Pa infection and chronic Pa infection incidence in a cohort of adult CF patients from a single centre. Polymerase chain reaction and direct sequencing were used to screen DNA samples for DCTN4 variants. A total of 121 adult CF patients from the Cochin Hospital CF centre have been included, all of them carrying two CFTR defects: 103 developed at least 1 pulmonary infection with Pa, and 68 patients of them had CPA. DCTN4 variants were identified in 24% (29/121) CF patients with Pa infection and in only 17% (3/18) CF patients with no Pa infection. Of the patients with CPA, 29% (20/68) had DCTN4 missense variants vs 23% (8/35) in patients without CPA. Interestingly, p.Tyr263Cys tend to be more frequently observed in CF patients with CPA than in patients without CPA (4/68 vs 0/35), and DCTN4 missense variants tend to be more frequent in male CF patients with CPA bearing two class II mutations than in male CF patients without CPA bearing two class II mutations (P = 0.06). Our observations reinforce that DCTN4 missense variants, especially p.Tyr263Cys, may be involved in the pathogenesis of CPA in male CF."
    ngrams = match(sentence)
    end = time.time()
    print(f"Time elapsed: {end-start:.3f} seconds")


