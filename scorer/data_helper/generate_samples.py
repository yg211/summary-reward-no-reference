from collections import OrderedDict
from nltk.tokenize import sent_tokenize
import itertools
import random
from tqdm import tqdm
import math
import numpy as np

from scorer.data_helper.json_reader import read_article_refs
from resources import LANGUAGE

def inHistory(history,swap):
    for entry in history:
        if set(swap) == set(entry):
            return True
    return False

def countAppearance(history,sub_seq):
    cnt = 0
    for entry in history:
        if set(sub_seq).issubset(set(entry)):
            cnt += 1
    return cnt

def getPossibleSwaps(pair_swaps, history, meta_history, n, article_sent_num):
    used_sent_in_ref = [ss[0] for ss in history]
    swaps = [pair for pair in pair_swaps if pair[0] not in used_sent_in_ref]

    poss_swaps = []
    if len(history) < n-1:
        for ss in swaps:
            cnt = countAppearance(meta_history,history+[ss])
            if cnt < math.pow(article_sent_num,n-1-len(history)):
                poss_swaps.append(ss)
    else:
        for ss in swaps:
            if not inHistory(meta_history,history+[ss]):
                poss_swaps.append(ss)

    if len(poss_swaps) == 0:
        print('error')
    return poss_swaps

def getSummaries(ref_sents, article_sents, n, num):
    if n > len(ref_sents):
        return []

    new_summaries = []
    pair_swaps = [pair for pair in itertools.product(range(len(ref_sents)), range(len(article_sents)))]
    ### compute number of all possible swaps: A_M^N * K^N, where A is permutation, M is num of sents in ref, K is
    ### num of sents in article, and N is num of sents to swap
    #nnum = math.factorial(len(ref_sents))/math.factorial(len(ref_sents)-n)
    #nnum *= math.pow(len(article_sents),n)
    #nnum = min(nnum/math.factorial(n)*0.8,num)

    meta_history = []
    break_flag = False
    while len(new_summaries)<num:
        pair_history = []
        summ = ref_sents[:]
        while len(pair_history) < n:
            possible_swaps = getPossibleSwaps(pair_swaps,pair_history,meta_history,n,len(article_sents))
            if len(possible_swaps) == 0:
                break_flag = True
                break
            pair = random.choice(possible_swaps)
            summ[pair[0]] = article_sents[pair[1]]
            pair_history.append(pair)
        if break_flag:
            break
        if ' '.join(summ) not in new_summaries:
            new_summaries.append(' '.join(summ))
            meta_history.append(pair_history)

    return new_summaries

def generateSampleSummaries(article_refs, n_list=[1], num=10):
    sample_summaries = OrderedDict()

    cnt = 0
    for entry in tqdm(article_refs):
        cnt += 1
        ref_sents = list(np.unique(sent_tokenize(entry['ref'],LANGUAGE)))
        article_sents = list(np.unique(sent_tokenize(entry['article'],LANGUAGE)))
        sample_summaries[entry['id']] = OrderedDict()

        for n in n_list:
            summaries = getSummaries(ref_sents, article_sents, n, num)
            sample_summaries[entry['id']][n] = summaries

    return sample_summaries


if __name__ == '__main__':
    article_refs = read_article_refs()
    sample_summaries = generateSampleSummaries(article_refs,n_list=[2],num=100)

