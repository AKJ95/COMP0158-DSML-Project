# Code from [1] is consulted and adopted.

import numpy as np


class VSM(object):

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

        self.load_txt(vecs_path, delimiter)

        if normalize:
            self.normalize()

    def load_txt(self, vecs_path, delimiter):
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
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar(self, vec, threshold=0.5, topn=10):
        sims = np.dot(self.vectors, vec).astype(self.dtype)
        sims_list = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            if sims_list[top_i] > threshold:
                r.append((self.labels[top_i], sims_list[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()


if __name__ == '__main__':
    p = 'data/processed/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
    vsm = VSM(p)
    print(f"VSM embedding stats: {vsm.vectors.shape}")
    print(f"VSM embedding example: {vsm.vectors[0][:5]}")
    print(f"VSM labels length: {len(vsm.labels)}")
    print(f"VSM label example: {vsm.labels[0]}")
