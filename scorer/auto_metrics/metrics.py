"""

"""
from nltk.tokenize import word_tokenize
from nltk.translate import bleu as _bleu
from nltk.translate.bleu_score import SmoothingFunction

#from scorer.auto_metrics.sent2vec_metric import Sent2Vec
from scorer.auto_metrics.meteor import Meteor
from scorer.auto_metrics.infersent_metric import InferSentScorer

chencherry = SmoothingFunction()
_meteor = None
def _m():
    global _meteor
    if _meteor is None:
        _meteor = Meteor()
    return _meteor

'''
_sent2vec = None
def _s():
    global _sent2vec
    if _sent2vec is None:
        _sent2vec = Sent2Vec()
    return _sent2vec
'''

def bleu(hyp, refs, n=2, smooth=True):
    # 1 - 4
    # nltk bleu
    smoothing_function = chencherry.method3 if smooth else chencherry.method0
    return _bleu(refs, hyp, [1./n for _ in range(n)], smoothing_function=smoothing_function)

def meteor(hyp, refs):
    # meteor package.
    return _m().score(hyp, refs)

#def sim(hyp, ref):
    #return _s().score(hyp, ref)

def test_metrics():
    hyps = ["Barack Obama will be the fourth president to receive the Nobel Peace Prize",
            "US President Barack Obama will fly to Oslo in Norway, for 26 hours and be the fourth US President in history to receive the Nobel Peace Prize."]
    refs = ["Barack Obama becomes the fourth American president to receive the Nobel Peace Prize", "The American president Barack Obama will fly into Oslo, Norway for 26 hours to receive the Nobel Peace Prize, the fourth American president in history to do so."]
    for hyp, ref in zip(hyps, refs):
        hyp_, ref_ = word_tokenize(hyp), word_tokenize(ref)
        for n in range(1,5):
            print("bleu{}: {:.5f}".format(n, bleu(hyp_, [ref_], n, smooth=False)))
        print("meteor: {:.5f}".format(meteor(hyp, [ref])))
        #print("sim: {:.3f}".format(sim(hyp, ref)))
        infers = InferSentScorer()
        print('infersent sim : {:.5f}'.format(infers(hyp,ref)))

if __name__ == '__main__':
    test_metrics()

