# Code from [1] is consulted and adopted.

import numpy as np


class VSM(object):
    """
    Vector space model representing the 1-NN classifier.
    """

    def __init__(self, vecs_path, dtype='float32', delimiter=' ', normalize=True):
        self.vecs_path = vecs_path

        if dtype == 'float32':
            self.dtype = np.float32
        elif dtype == 'float16':
            self.dtype = np.float16
        else:
            self.dtype = np.float

        self.labels = []
        self.vectors = np.array([], dtype=self.dtype)
        self.indices = {}
        self.ndims = 0

        # Loads pre-computed entity embeddings.
        self.load_txt(vecs_path, delimiter)

        if normalize:
            self.normalize()

    def load_txt(self, vecs_path, delimiter):
        """
        Loads pre-computed entity embeddings from the designated file.
        :param vecs_path: Filepath to the pre-computed entity embeddings.
        :param delimiter: Delimiter separating contents in a line.
        """
        self.vectors = []
        with open(vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.strip().split(delimiter)
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=self.dtype))

        self.vectors = np.vstack(self.vectors)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def save_txt(self, vecs_path, delimiter='\t'):
        with open(vecs_path, 'w') as vecs_f:
            for label, vec in zip(self.labels, self.vectors):
                vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
                vecs_f.write('%s%s%s\n' % (label, delimiter, vec_str))

    def normalize(self):
        self.vectors = (self.vectors.T / np.linalg.norm(self.vectors, axis=1)).T

    def get_vec(self, label):
        """
        Get entity embedding via label, aka CUI.
        :param label: CUI in this case.
        :return: Pre-computed entity embedding corresponding to the given CUI.
        """
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        """
        Consine similarity between the entity embeddings of the two given CUIs.
        :param label1: CUI 1
        :param label2: CUI 2
        :return: Cosine similarity between the two entity embeddings.
        """
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar(self, vec, threshold=0.5, topn=10):
        """
        Finds the top-n most similar entities to the given mention embedding above the given threshold.
        """
        sims = np.dot(self.vectors, vec).astype(self.dtype)
        sims_list = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            if sims_list[top_i] > threshold:
                r.append((self.labels[top_i], sims_list[top_i]))
        return r

    # def sims(self, vec):
    #     return np.dot(self.vectors, np.array(vec)).tolist()
