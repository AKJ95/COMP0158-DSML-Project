# Code from [1] is consulted and adopted.

import joblib

import numpy as np
import torch

from pytt_hf import toks2vecs
from matcher_simstring import SimString_UMLS
from vectorspace import VSM
from NERComponent import NERComponent
from softmax_pytorch import SoftmaxClassifier


# Helper normalisation function
def norm(v):
    """
    Normalise a vector
    :param v: vector
    :return: normalised vector
    """
    return v / np.linalg.norm(v)


# Helper class representing a document with its detected spans
class MedLinkerDoc(object):

    def __init__(self, text, tokens, spans):
        #
        self.text = text
        self.tokens = tokens
        self.spans = spans
        self.vectors = []

    def set_contextual_vectors(self):
        # Set contextual embeddings for all its tokens
        self.vectors = toks2vecs(self.tokens, return_tokens=False)
        assert len(self.vectors) == len(self.tokens)

    def get_span_vector(self, span_start, span_end, normalize=False):
        # Get the contextual embeddings for a given span by averaging the embeddings of the tokens within the span.
        span_vecs = [self.vectors[i] for i in range(span_start, span_end)]
        span_vec = np.array(span_vecs).mean(axis=0)

        if normalize:
            span_vec = norm(span_vec)

        return span_vec

    def get_spans(self, include_vectors=True, normalize=False):
        # Returns all spans in the document with their vectors if requested.
        output_spans = []

        for span_start, span_end in self.spans:
            if include_vectors and len(self.vectors) > 0:
                span_vec = self.get_span_vector(span_start, span_end, normalize=normalize)
                output_spans.append((span_start, span_end, span_vec))
            else:
                output_spans.append((span_start, span_end))

        return output_spans


# Full MedLinker pipeline
class MedLinker(object):
    """
    Full MedLinker pipeline
    """

    def __init__(self, medner, umls_kb):
        #
        self.medner = medner
        self.umls_kb = umls_kb

        self.string_matcher = None
        self.exact_matcher = None
        self.sty_clf = None
        self.cui_clf = None

        self.st_vsm = None
        self.cui_vsm = None

        self.cui_validator = None
        self.cui_validator_thresh = None

        self.st_validator = None
        self.st_validator_thresh = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_cui_softmax_pt(self):
        """
        Load the MLP classifier.
        """
        self.cui_clf = SoftmaxClassifier(18426,
                                         'models/Classifiers/softmax.cui.map',
                                         'models/Classifiers/softmax.cui.pt')
        self.cui_clf.to(self.device)

    def load_st_VSM(self, st_vsm_path):
        #
        self.st_vsm = VSM(st_vsm_path)

    def load_cui_VSM(self, cui_vecs_path):
        """
        Load the 1-NN classifier.
        :param cui_vecs_path: Filepath to the pre-computed entity embeddings.
        """
        self.cui_vsm = VSM(cui_vecs_path)

    def load_cui_validator(self, clf_path, validator_thresh=0.5):
        """
        Load the logistic regression meta learner.
        :param clf_path: Model checkpoint of the meta learner.
        :param validator_thresh: Decision threshold.
        """
        #
        self.cui_validator = joblib.load(clf_path)
        self.cui_validator_thresh = validator_thresh

    def load_string_matcher(self, ngram_db_path, ngram_map_path):
        """
        Load the string matcher.
        :param ngram_db_path: Filepath to database matching entity aliases/synonyms to their n-gram features.
        :param ngram_map_path: Filepath to mapping file of entity aliases/synonyms to their CUIs.
        """
        self.string_matcher = SimString_UMLS(self.umls_kb, ngram_db_path, ngram_map_path)

    def predict(self, sentence,
                gold_tokens=None, gold_spans=None,
                predict_cui=True, predict_sty=False,
                require_cui=True, require_sty=False,
                top_n=1):
        """
        Predict CUIs given a sentence with detected spans or gold spans. Can return multiple predictions depending on
        the top_n parameter.
        """

        # Get the detected spans using the MD component or the gold spans of the sentence
        if (gold_tokens is not None) and (gold_spans is not None):
            tokens, spans = gold_tokens, gold_spans
        else:
            ner_prediction = self.medner.predict(sentence)
            tokens = ner_prediction.tokens
            spans = []
            for span in ner_prediction.spans:
                spans.append((span.start, span.end))

        # Create the helper MedLinkerDoc object, containing all sentences in the document with their spans.
        doc = MedLinkerDoc(sentence, tokens, spans)
        doc.set_contextual_vectors()

        r = {'sentence': sentence, 'tokens': tokens, 'spans': []}

        # Iterate through each span in the document and predict their CUIs.
        for span_start, span_end, span_vec in doc.get_spans(include_vectors=True, normalize=False):
            span_str = ' '.join(doc.tokens[span_start:span_end])
            span_info = {'start': span_start, 'end': span_end, 'text': span_str, 'st': None, 'cui': []}

            # Predict the CUI for the current span
            if predict_cui:
                span_cuis = self.match_cui(span_str, span_vec)

                if span_cuis is not None:
                    # span_info['cui'] = span_cuis[0]
                    for i in range(min(top_n, len(span_cuis))):
                        span_info['cui'].append(span_cuis[i])
                elif require_cui:
                    continue

            if predict_sty:
                span_sts = self.match_sty(span_str, span_vec)

                if span_sts is not None:
                    span_info['st'] = span_sts[0]
                elif require_sty:
                    continue

            r['spans'].append(span_info)

        return r

    def match_sty(self, span_str, span_ctx_vec):
        #
        matches_str = []
        if self.string_matcher is not None:
            matches_str = self.string_matcher.match_sts(span_str.lower())
        elif self.exact_matcher is not None:
            matches_str = self.exact_matcher.match_sts(span_str.lower())
            matches_str = [(st, 1 / (1 + idx)) for idx, (st, _, _, _) in enumerate(matches_str)]

        matches_ctx = []
        if self.sty_clf is not None:
            matches_ctx = self.sty_clf.predict(span_ctx_vec)
        elif self.st_vsm is not None:
            span_ctx_vec = norm(span_ctx_vec)
            matches_ctx = self.st_vsm.most_similar(span_ctx_vec, threshold=0.5)

        scores_str, scores_ctx = dict(matches_str), dict(matches_ctx)
        matches = {sty: max(scores_str.get(sty, 0), scores_ctx.get(sty, 0))
                   for sty in scores_str.keys() | scores_ctx.keys()}
        matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

        if (self.st_validator is not None) and (len(matches) > 0):
            pred_valid = self.validate_st_pred(matches_str, scores_str,
                                               matches_ctx, scores_ctx,
                                               matches)

            if pred_valid == False:
                matches = {}

        if len(matches) > 0:
            return matches
        else:
            return None

    def match_cui(self, span_str, span_ctx_vec):
        # Returns all entities matched by the string matcher with their scores. (Above the 0.5 score threshold)
        matches_str = []
        if self.string_matcher is not None:
            matches_str = self.string_matcher.match_cuis(span_str.lower())
        elif self.exact_matcher is not None:
            matches_str = self.exact_matcher.match_cuis(span_str.lower())
            matches_str = [(cui, 1 / (1 + idx)) for idx, (cui, _, _, _) in enumerate(matches_str)]

        matches_ctx = []
        vsm_matches_ctx = []

        # Returns all entities matched by the MLP classifier with their scores. (Above the 0.5 score threshold)
        if self.cui_clf is not None:
            span_ctx_vec_tensor = torch.unsqueeze(torch.from_numpy(span_ctx_vec), 0)
            span_ctx_vec_tensor = span_ctx_vec_tensor.to(self.device)
            matches_ctx = self.cui_clf.predict(span_ctx_vec_tensor)

        # Returns all entities matched by the 1-NN classifier with their scores. (Above the 0.5 score threshold)
        if self.cui_vsm is not None:
            span_ctx_vec = norm(span_ctx_vec)
            vsm_matches_ctx = self.cui_vsm.most_similar(span_ctx_vec, threshold=0.5, topn=100)

        scores_str, scores_ctx, scores_vsm = dict(matches_str), dict(matches_ctx), dict(vsm_matches_ctx)

        # Combine the scores from all three classifiers with some ensemble scheme. Comment and uncomment to choose the
        # ensemble scheme wanted.

        # This scheme is the String Matcher + 1-NN Classifier + MLP Classifier (averag method)
        # matches = {cui: max(scores_str.get(cui, 0), (scores_ctx.get(cui, 0) + scores_vsm.get(cui, 0)) / 2)
        #            for cui in scores_str.keys() | scores_ctx.keys() | scores_vsm.keys()}

        # This scheme is used by all other configurations, which scores entity by the maximum score from any one of the
        # classifiers that are used in the configuration.
        matches = {cui: max(scores_str.get(cui, 0), scores_ctx.get(cui, 0), scores_vsm.get(cui, 0))
                   for cui in scores_str.keys() | scores_ctx.keys() | scores_vsm.keys()}

        # Rank all matches
        matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        if (self.cui_validator is not None) and (len(matches) > 0):
            pred_valid = self.validate_cui_pred(matches_str, scores_str,
                                                matches_ctx, scores_ctx,
                                                matches)

            if pred_valid == False:
                matches = {}

        # Return all matches
        if len(matches) > 0:
            return matches
        else:
            return None

    def validate_cui_pred(self, matches_str, scores_str, matches_vsm, scores_vsm, matches_joint):
        """
        Accept or reject the CUI prediction based on the meta learner.
        """

        matchers_agree = False
        if len(matches_str) > 0 and len(matches_vsm) > 0:
            if matches_str[0][0] == matches_vsm[0][0]:
                matchers_agree = True

        top_match = matches_joint[0][0]

        x = []
        if len(matches_str) > 0:
            x.append(matches_str[0][1])
        else:
            x.append(0)
        if len(matches_vsm) > 0:
            x.append(matches_vsm[0][1])
        else:
            x.append(0)
        x.append(matches_joint[0][1])
        x.append((scores_str.get(top_match, 0) + scores_vsm.get(top_match, 0)) / 2)
        x.append(int(matchers_agree))

        prob_F, prob_T = self.cui_validator.predict_proba([x])[0]

        if prob_T >= self.cui_validator_thresh:
            return True
        else:
            return False

    def validate_st_pred(self, matches_str, scores_str, matches_vsm, scores_vsm, matches_joint):

        matchers_agree = False
        if len(matches_str) > 0 and len(matches_vsm) > 0:
            if matches_str[0][0] == matches_vsm[0][0]:
                matchers_agree = True

        top_match = matches_joint[0][0]

        x = []
        if len(matches_str) > 0:
            x.append(matches_str[0][1])
        else:
            x.append(0)
        if len(matches_vsm) > 0:
            x.append(matches_vsm[0][1])
        else:
            x.append(0)
        x.append(matches_joint[0][1])
        x.append((scores_str.get(top_match, 0) + scores_vsm.get(top_match, 0)) / 2)
        x.append(int(matchers_agree))

        prob_F, prob_T = self.st_validator.predict_proba([x])[0]

        if prob_T >= self.st_validator_thresh:
            return True
        else:
            return False


if __name__ == '__main__':
    pass
