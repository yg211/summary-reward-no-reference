import numpy as np
import os
from nltk import PorterStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from pytorch_transformers import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse

from scorer.data_helper.json_reader import read_sorted_scores, read_articles, read_processed_scores, read_scores
from helpers.data_helpers import sent2stokens_wostop, sent2tokens_wostop, sent2stokens, text_normalization
from scorer.auto_metrics.metrics import bleu, meteor
from resources import RUNS_DIR, ROUGE_DIR, BASE_DIR, MODEL_WEIGHT_DIR
from scorer.auto_metrics.rouge.rouge import RougeScorer
from step1_encode_doc_summ import raw_bert_encoder
from rewarder import Rewarder

def sts_bert_encoder(model, sent_list):
    if not isinstance(sent_list,list):
        assert isinstance(sent_list,str)
        sent_list = sent_tokenize(sent_list)
    vecs = model.encode(sent_list)
    return vecs


def sts_bert_rewarder(model, text1, text2):
    vec_list1 = sts_bert_encoder(model,text1)
    vec_list2 = sts_bert_encoder(model,text2)
    avg_vec1 = np.mean(vec_list1,axis=0)
    avg_vec2 = np.mean(vec_list2,axis=0)
    return cosine_similarity(avg_vec1.reshape(1, -1), avg_vec2.reshape(1, -1))[0][0]


def raw_bert_rewarder(model, tokenizer, text1, text2):
    v1 = raw_bert_encoder(model,tokenizer,[text1])
    v2 = raw_bert_encoder(model,tokenizer,[text2])
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]


def evaluate_metric(metric, stem, remove_stop, with_ref, prompt='overall'):
    ''' metrics that use reference summaries '''
    assert metric in ['ROUGE-1-F', 'ROUGE-1-R', 'ROUGE-2-F', 'ROUGE-2-R', 'ROUGE-L-F', 'ROUGE-L-R', 'ROUGE-SU*-F',
                      'ROUGE-SU*-R', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-5', 'meteor',
                      'infersent', 'bert-raw','bert-sts','bert-nli','bert-human','mover-1', 'mover-2', 'mover-smd']
    stemmed_str = "_stem" if stem else ""
    stop_str = "_removestop" if remove_stop else ""
    if with_ref:
        ranks_file_path = os.path.join('outputs', 'wref_{}{}{}_{}_rank_correlation.csv'.format(metric, stemmed_str, stop_str, prompt))
    else:
        ranks_file_path = os.path.join('outputs', 'woref_{}{}{}_{}_rank_correlation.csv'.format(metric, stemmed_str, stop_str, prompt))
    print('\n====={}=====\n'.format(ranks_file_path))

    #if os.path.isfile(ranks_file_path):
        #return ranks_file_path

    ranks_file = open(ranks_file_path, 'w')
    ranks_file.write('article,summ_id,human_score,metric_score\n')

    sorted_scores = read_sorted_scores()
    input_articles, _ = read_articles()
    corr_data = np.zeros((len(sorted_scores), 3))

    stopwords_list = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    if metric.startswith('infersent'):
        from scorer.auto_metrics.infersent_metric import InferSentScorer
        infers = InferSentScorer()
    elif metric.startswith('sent2vec'):
        from scorer.auto_metrics.sent2vec_metric import Sent2Vec
        s2v = Sent2Vec()
    elif metric.startswith('bert'):
        pass
        if 'human' in metric:
            rewarder = Rewarder(os.path.join(MODEL_WEIGHT_DIR,'sample.model'))
        elif 'sts' in metric:
            bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        elif 'nli' in metric:
            bert_model = SentenceTransformer('bert-large-nli-mean-tokens')
        else:
            #raw BERT
            bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            bert_model = BertModel.from_pretrained('bert-large-uncased')
    elif metric.startswith('mover'):
        print('Make sure that your have started the mover server. Find details at https://github.com/AIPHES/emnlp19-moverscore.')
        from summ_eval.client import EvalClient
        mover_scorer = EvalClient()

    for i, (article_id, scores_list) in tqdm(enumerate(sorted_scores.items())):
        human_ranks = [s['scores'][prompt] for s in scores_list]
        if len(human_ranks) < 2: continue
        ref_summ = scores_list[0]['ref']
        article = [entry['article'] for entry in input_articles if entry['id']==article_id][0]

        if stem and remove_stop:
            sys_summs = [" ".join(sent2stokens_wostop(s['sys_summ'], stemmer, stopwords_list, 'english', True)) for s in scores_list]
            ref_summ = " ".join(sent2stokens_wostop(ref_summ, stemmer, stopwords_list, 'english', True))
            article = " ".join(sent2stokens_wostop(article, stemmer, stopwords_list, 'english', True))
        elif not stem and remove_stop:
            sys_summs = [" ".join(sent2tokens_wostop(s['sys_summ'], stopwords_list, 'english', True)) for s in scores_list]
            ref_summ = " ".join(sent2tokens_wostop(ref_summ, stopwords_list, 'english', True))
            article = " ".join(sent2tokens_wostop(article, stopwords_list, 'english', True))
        elif not remove_stop and stem:
            sys_summs = [" ".join(sent2stokens(s['sys_summ'], stemmer, 'english', True)) for s in scores_list]
            ref_summ = " ".join(sent2stokens(ref_summ, stemmer, 'english', True))
            article = " ".join(sent2stokens(article, stemmer, 'english', True))
        else:
            sys_summs = [s['sys_summ'] for s in scores_list]

        summ_ids = [s['summ_id'] for s in scores_list]
        sys_summs = [text_normalization(s) for s in sys_summs]
        ref_summ = text_normalization(ref_summ)
        article = text_normalization(article)

        if 'rouge' in metric.lower():
            auto_metric_ranks = []
            for ss in sys_summs:
                rouge_scorer = RougeScorer(ROUGE_DIR,BASE_DIR)
                if with_ref: auto_metric_ranks.append(rouge_scorer(ss, ref_summ)[metric])
                else: auto_metric_ranks.append(rouge_scorer(ss, article)[metric])
        elif metric.startswith('bleu'):
            n = int(metric.split('-')[1])
            if with_ref: auto_metric_ranks = [bleu(ss, [ref_summ], n, smooth=False) for ss in sys_summs]
            else:  auto_metric_ranks = [bleu(ss, [article], n, smooth=False) for ss in sys_summs]
        elif metric.startswith('meteor'):
            if with_ref: auto_metric_ranks = [meteor(ss, [ref_summ]) for ss in sys_summs]
            else: auto_metric_ranks = [meteor(ss, [article]) for ss in sys_summs]
        elif metric.startswith('infersent'):
            if with_ref: auto_metric_ranks = [infers(ss, ref_summ) for ss in sys_summs]
            else: auto_metric_ranks = [infers(ss, article) for ss in sys_summs]
        elif metric.startswith('sent2vec'):
            if with_ref: auto_metric_ranks = [s2v.score(ss, ref_summ) for ss in sys_summs]
            else: auto_metric_ranks = [s2v.score(ss, article) for ss in sys_summs]
        elif metric.startswith('bert'):
            if 'human' in metric:
                if with_ref: auto_metric_ranks = [rewarder(ref_summ,ss) for ss in sys_summs]
                else: auto_metric_ranks = [rewarder(article,ss) for ss in sys_summs]
            elif 'sts' in metric or 'nli' in metric:
                if with_ref: auto_metric_ranks = [sts_bert_rewarder(bert_model,ss,ref_summ) for ss in sys_summs]
                else: auto_metric_ranks = [sts_bert_rewarder(bert_model,ss,article) for ss in sys_summs]
            else: #raw BERT encoder
                if with_ref: auto_metric_ranks = [raw_bert_rewarder(bert_model,bert_tokenizer,ss,ref_summ) for ss in sys_summs]
                else: auto_metric_ranks = [raw_bert_rewarder(bert_model,bert_tokenizer,ss,article) for ss in sys_summs]
        elif metric.startswith('mover'):
            if '1' in metric: mm = 'wmd_1'
            elif '2' in metric: mm = 'wmd_2'
            else: mm = 'smd'
            if with_ref: cases = [ [[ss], [ref_summ], mm] for ss in sys_summs ]
            else: cases = [ [[ss], sent_tokenize(article), mm] for ss in sys_summs ]
            auto_metric_ranks = mover_scorer.eval(cases)['0']

        for sid, amr, hr in zip(summ_ids, auto_metric_ranks, human_ranks):
            ranks_file.write('{},{},{:.2f},{:.4f}\n'.format(article_id, sid, hr, amr))

        spearmanr_result = spearmanr(human_ranks, auto_metric_ranks)
        print(spearmanr_result[0])
        pearsonr_result = pearsonr(human_ranks, auto_metric_ranks)
        kendalltau_result = kendalltau(human_ranks, auto_metric_ranks)
        corr_data[i, :] = [spearmanr_result[0], pearsonr_result[0], kendalltau_result[0]]

    corr_mean_all = np.nanmean(corr_data, axis=0)
    print('\n====={}=====\n'.format(ranks_file_path))
    print("Correlation mean on all data spearman/pearsonr/kendall: {}".format(corr_mean_all))

    ranks_file.flush()
    ranks_file.close()

    return ranks_file_path

def parse_args():
    ap = argparse.ArgumentParser("arguments for summary sampler")
    ap.add_argument('-m','--metric',type=str,default='mover-1',choices=['ROUGE-1-F', 'ROUGE-1-R', 'ROUGE-2-F', 'ROUGE-2-R', 'ROUGE-L-F', 'ROUGE-L-R', 'ROUGE-SU*-F',
                      'ROUGE-SU*-R', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-5', 'meteor',
                      'infersent', 'bert-raw','bert-sts','bert-nli','bert-human', 'mover-1', 'mover-2', 'mover-smd'],help='compare which metric against the human judgements')
    ap.add_argument('-p','--prompt',type=str,default='overall',help='which human ratings you want to use as ground truth',choices=['overall','grammar'])
    ap.add_argument('-r','--with_ref',type=int,default=0,help='whether to use references in your metric; 1: yes, 0: no')
    ap.add_argument('-s','--stem',type=int,help='whether stem the texts before computing the metrics; 1 yes, 0 no')
    ap.add_argument('-rs','--remove_stop',type=int,help='whether remove stop words in texts before computing the metrics; 1 yes, 0 no')
    args = ap.parse_args()
    return args.metric, args.prompt, args.with_ref, args.stem, args.remove_stop


if __name__ == '__main__':
    metric, prompt, with_ref, stem, remove_stop = parse_args()
    with_ref = bool(with_ref)
    stem = bool(stem)
    remove_stop = bool(remove_stop)

    print('\n=====Arguments====')
    print('metric: '+metric)
    print('prompt: '+prompt)
    print('with ref: '+repr(with_ref))
    print('stem: '+repr(stem))
    print('remove stopwords: '+repr(remove_stop))
    print('=====Arguments====\n')

    metric_scores_file = evaluate_metric(metric,stem,remove_stop,with_ref,prompt)
