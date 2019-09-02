import numpy as np
import os
from nltk import PorterStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import csv
import torch
from pytorch_transformers import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from scorer.data_helper.json_reader import read_sorted_scores, read_articles, read_processed_scores, read_scores
from helpers.data_helpers import sent2stokens_wostop, sent2tokens_wostop, sent2stokens, text_normalization
from scorer.auto_metrics.metrics import bleu, meteor
from resources import RUNS_DIR, ROUGE_DIR, BASE_DIR
from scorer.auto_metrics.rouge.rouge import RougeScorer


def raw_bert_encoder(model, tokenizer, sent_list, stride=128, gpu=True):
    merged_text = ''
    for ss in sent_list: merged_text += ss+' '
    tokens = tokenizer.encode(merged_text)
    #print(len(tokens))

    model.eval()
    with torch.no_grad():
        if len(tokens) <= 510:
            tokens = torch.tensor(tokens).unsqueeze(0)
            if gpu:
                tokens = tokens.to('cuda')
                model.to('cuda')
            vv = model(tokens)[0][0].data.cpu().numpy()
            vv = np.mean(vv,axis=0)
        else:
            end_pointer = stride
            batch = []
            real_length = []
            att_masks = []
            while True:
                start_pointer = end_pointer-510
                if start_pointer < 0: start_pointer = 0
                if start_pointer >= len(tokens): break
                if end_pointer <= len(tokens):
                    batch.append(tokens[start_pointer:end_pointer])
                    real_length.append(end_pointer-start_pointer)
                    att_masks.append([1]*real_length[-1])
                else:
                    batch.append(tokens[start_pointer:end_pointer])
                    real_length.append(len(tokens)-start_pointer)
                    att_masks.append([1] * real_length[-1])
                end_pointer += stride
                #print(len(batch[-1]))

            #padding
            longest = max(real_length)
            for ii in range(len(batch)):
                batch[ii] += [0] * (longest-real_length[ii])
                att_masks[ii] += [0] * (longest-real_length[ii])

            batch = torch.tensor(batch)
            att_masks = torch.tensor(att_masks)
            if gpu:
                batch = batch.to('cuda')
                att_masks = att_masks.to('cuda')
                model.to('cuda')

            last_layers = model(input_ids=batch,attention_mask=att_masks)[0].data.cpu().numpy()
            vectors = []
            for ii,bb in enumerate(last_layers):
                vectors.append(np.mean(bb[:real_length[ii]],axis=0))
            vv = np.mean(vectors,axis=0)

    return vv


def encode_doc_summ(stem=False, stop=False):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert_model = BertModel.from_pretrained('bert-large-uncased')

    sorted_scores = read_sorted_scores()
    input_articles, _ = read_articles()

    stopwords_list = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    vec_dic = {}

    for i, (article_id, scores_list) in tqdm(enumerate(sorted_scores.items())):
        vec_dic[article_id] = {}
        if stem and stop:
            sys_summs = [" ".join(sent2stokens_wostop(s['sys_summ'], stemmer, stopwords_list, 'english', True)) for s in scores_list]
            ref_summs = [" ".join(sent2stokens_wostop(s['ref'], stemmer, stopwords_list, 'english', True)) for s in scores_list]
        elif not stem and stop:
            sys_summs = [" ".join(sent2tokens_wostop(s['sys_summ'], stopwords_list, 'english', True)) for s in scores_list]
            ref_summs = [" ".join(sent2tokens_wostop(s['ref'], stopwords_list, 'english', True)) for s in
                         scores_list]
        elif not stop and stem:
            sys_summs = [" ".join(sent2stokens(s['sys_summ'], stemmer, 'english', True)) for s in
                         scores_list]
            ref_summs = [" ".join(sent2stokens(s['ref'], stemmer, 'english', True)) for s in
                     scores_list]
        else:
            sys_summs = [s['sys_summ'] for s in scores_list]
            ref_summs = [s['ref'] for s in scores_list]

        summ_ids = [s['summ_id'] for s in scores_list]
        article = [entry['article'] for entry in input_articles if entry['id']==article_id][0]

        # clean text
        sys_summs = [text_normalization(s) for s in sys_summs]
        ref_summs = [text_normalization(s) for s in ref_summs]
        article = text_normalization(article)

        '''
        print('\n===article===')
        print(article)
        print('\n===ref summ===')
        print(ref_summs)
        for i,sid in enumerate(summ_ids):
            print('\n===sys summ {}==='.format(sid))
            print(sys_summs[i])
        '''

        vec_dic[article_id]['article'] = raw_bert_encoder(bert_model, bert_tokenizer, [article])
        vec_dic[article_id]['ref'] = raw_bert_encoder(bert_model, bert_tokenizer, [ref_summs[0]])
        for i,sid in enumerate(summ_ids):
            vec_dic[article_id]['sys_summ{}'.format(sid)] = raw_bert_encoder(bert_model, bert_tokenizer, [sys_summs[i]])

    with open('data/raw_bert_vectors.p','wb') as ff:
        pickle.dump(vec_dic,ff)


if __name__ == '__main__':
    encode_doc_summ()
