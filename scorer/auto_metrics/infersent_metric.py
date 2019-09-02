from resources import INFERSENT_PATH, W2V_PATH, VEC_DIM
from scorer.auto_metrics.infersent_model import InferSent
import torch
import numpy as np

def _norm(x):
    z = np.linalg.norm(x)
    return x/z if z > 1e-10 else 0 * x

class InferSentScorer:
    def __init__(self,gpu=False):
        self.version = 1
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': VEC_DIM,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': self.version}
        self.infersent = InferSent(params_model)
        if gpu:
            self.infersent = self.infersent.cuda(0)
        self.infersent.load_state_dict(torch.load(INFERSENT_PATH))
        self.infersent.set_w2v_path(W2V_PATH)
        self.infersent.build_vocab_k_words(K=100000)


    def __call__(self, sent1, sent2):
        emb1 = self.getEmbedding([sent1])[0]
        emb2 = self.getEmbedding([sent2])[0]
        return np.dot(_norm(emb1), _norm(emb2))

    def getEmbedding(self, sentences):
        return self.infersent.encode(sentences, tokenize=True)
