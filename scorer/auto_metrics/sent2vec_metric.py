#!/usr/bin/env python

# Python wrapper for METEOR implementation

import os
import pexpect
import numpy as np
import sys
import sent2vec

from resources import SENT2VEC_DIR

SENT2VEC_MODEL = os.environ.get('SENT2VEC_MODEL', os.path.join(SENT2VEC_DIR, 'wiki_bigrams.bin'))


def _norm(x):
    z = np.linalg.norm(x)
    return x / z if z > 1e-10 else 0 * x


class Sent2Vec:
    def __init__(self):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(SENT2VEC_MODEL)

    def embed(self, x):
        vec = self.model.embed_sentence(x)
        return np.squeeze(np.asarray(vec))

    def score(self, x, y):
        try:
            return _norm(self.embed(x)).dot(_norm(self.embed(y)))
        except pexpect.TIMEOUT:
            sys.stderr.write("Timed out while computing: {} {}".format(x, y))
            return 0.


def test_sent2vec():
    x = "Barack Obama will be the fourth president to receive the Nobel Peace Prize"
    y = "Barack Obama becomes the fourth American president to receive the Nobel Peace Prize"
    s2v = Sent2Vec()
    print(s2v.score(x, y))


if __name__ == '__main__':
    test_sent2vec()
