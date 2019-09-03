import torch
from torch.autograd import Variable
import numpy as np
import os
import argparse
import random
import copy
from tqdm import tqdm
import pickle
from scorer.data_helper.json_reader import read_sorted_scores, read_articles, read_processed_scores, read_scores
from scipy.stats import spearmanr, pearsonr, kendalltau
import math

from resources import MODEL_WEIGHT_DIR


def parse_split_data(sorted_scores, train_percent, dev_percent, prompt='overall'):
    train = {}
    dev = {}
    test = {}
    all = {}

    for article_id, scores_list in tqdm(sorted_scores.items()):
        entry = {}
        summ_ids = [s['summ_id'] for s in scores_list]
        for sid in summ_ids:
            entry['sys_summ'+repr(sid)] = [s['scores'][prompt] for s in scores_list if s['summ_id']==sid][0]

        rand = random.random()
        all[article_id] = entry
        if rand < train_percent : train[article_id] = entry
        elif rand < train_percent+dev_percent : dev[article_id] = entry
        else: test[article_id] = entry

    return train, dev, test, all


def build_model(model_type, vec_length, learn_rate=None):
    if 'linear' in model_type:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, 1),
        )
    else:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, int(vec_length/2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(vec_length/2), 1),
        )
    if learn_rate is not None:
        optimiser = torch.optim.Adam(deep_model.parameters(),lr=learn_rate)
        return deep_model, optimiser
    else:
        return deep_model


def deep_pair_train(vec_list, target, deep_model, optimiser, device):
    input = Variable(torch.from_numpy(np.array(vec_list)).float())
    if 'gpu' in device:
        input = input.to('cuda')
    value_variables = deep_model(input)
    softmax_layer = torch.nn.Softmax(dim=1)
    pred = softmax_layer(value_variables)
    target_variables = Variable(torch.from_numpy(np.array(target)).float()).view(-1,2,1)
    if 'gpu' in device:
        target_variables = target_variables.to('cuda')

    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(pred,target_variables)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.cpu().item()


def build_pairs(entries):
    pair_list = []
    for article_id in entries:
        entry = entries[article_id]
        summ_ids = list(entry.keys())
        for i in range(len(summ_ids)-1):
            for j in range(1, len(summ_ids)):
                if entry[summ_ids[i]] > entry[summ_ids[j]]: pref = [1,0]
                elif entry[summ_ids[i]] < entry[summ_ids[j]]: pref = [0,1]
                else: pref = [0.5, 0.5]
                pair_list.append( (article_id,summ_ids[i],summ_ids[j],pref) )

    return pair_list

        
def build_pair_vecs(vecs, pairs):
    pair_vec_list = []
    for aid, sid1, sid2, _ in pairs:
        article_vec = list(vecs[aid]['article'])
        s1_vec = list(vecs[aid][sid1])
        s2_vec = list(vecs[aid][sid2])
        pair_vec_list.append([article_vec+s1_vec, article_vec+s2_vec])
    return pair_vec_list


def pair_train_rewarder(vec_dic, pairs, deep_model, optimiser, batch_size=32, device='cpu'):
    loss_list = []
    shuffled_pairs = pairs[:]
    np.random.shuffle(shuffled_pairs)
    vec_pairs = build_pair_vecs(vec_dic, shuffled_pairs)
    #print('total number of pairs built: {}'.format(len(vec_pairs)))

    for pointer in range(int(len(pairs)/batch_size)+1):
        vec_batch = vec_pairs[pointer*batch_size:(pointer+1)*batch_size]
        target_batch = shuffled_pairs[pointer*batch_size:(pointer+1)*batch_size]
        target_batch = [ee[-1] for ee in target_batch]
        loss = deep_pair_train(vec_batch,target_batch,deep_model,optimiser,device)
        loss_list.append(loss)

    return np.mean(loss_list)


def test_rewarder(vec_list, human_scores, model, device):
    results = {'rho':[], 'pcc':[], 'tau':[]}
    for article_id in human_scores:
        entry = human_scores[article_id]
        summ_ids = list(entry.keys())
        if len(summ_ids) < 2: continue
        concat_vecs = []
        true_scores = []
        for i in range(len(summ_ids)):
            article_vec = list(vec_list[article_id]['article'])
            summ_vec = list(vec_list[article_id][summ_ids[i]])
            concat_vecs.append(article_vec+summ_vec)
            true_scores.append(entry[summ_ids[i]])
        input = Variable(torch.from_numpy(np.array(concat_vecs)).float())
        if 'gpu' in device:
            input = input.to('cuda')
        model.eval()
        with torch.no_grad():
            pred_scores = model(input).data.cpu().numpy().reshape(1,-1)[0]

        rho = spearmanr(true_scores, pred_scores)[0]
        pcc = pearsonr(true_scores, pred_scores)[0]
        tau = kendalltau(true_scores, pred_scores)[0]
        if not(math.isnan(rho) or math.isnan(pcc) or math.isnan(tau)):
            results['rho'].append(rho)
            results['pcc'].append(pcc)
            results['tau'].append(tau)

    return results


def parse_args():
    ap = argparse.ArgumentParser("arguments for summary sampler")
    ap.add_argument('-e','--epoch_num',type=int,default=50)
    ap.add_argument('-b','--batch_size',type=int,default=32)
    ap.add_argument('-tt','--train_type',type=str,help='pairwise or regression', default='pairwise')
    ap.add_argument('-tp','--train_percent',type=float,help='how many data used for training', default=.64)
    ap.add_argument('-dp','--dev_percent',type=float,help='how many data used for dev', default=.16)
    ap.add_argument('-lr','--learn_rate',type=float,help='learning rate', default=3e-4)
    ap.add_argument('-mt','--model_type',type=str,help='deep/linear', default='linear')
    ap.add_argument('-dv','--device',type=str,help='cpu/gpu', default='gpu')

    args = ap.parse_args()
    return args.epoch_num, args.batch_size, args.train_type, args.train_percent, args.dev_percent, args.learn_rate, args.model_type, args.device


if __name__ == '__main__':
    epoch_num, batch_size, train_type, train_percent, dev_percent, learn_rate, model_type, device = parse_args()

    print('\n=====Arguments====')
    print('epoch num {}'.format(epoch_num))
    print('batch size {}'.format(batch_size))
    print('train type {}'.format(train_type))
    print('train percent {}'.format(train_percent))
    print('dev percent {}'.format(dev_percent))
    print('learn rate {}'.format(learn_rate))
    print('model type {}'.format(model_type))
    print('device {}'.format(device))
    print('=====Arguments====\n')

    if train_percent + dev_percent >= 1.:
        print('ERROR! Train data percentage plus dev data percentage is {}! Make sure the sum is below 1.0!'.format(train_percent+dev_percent))
        exit(1)

    BERT_VEC_LENGTH = 1024 # change this to 768 if you use bert-base
    deep_model, optimiser = build_model(model_type,BERT_VEC_LENGTH*2,learn_rate)
    if 'gpu' in device:
        deep_model.to('cuda')

    # read human scores and vectors for summaries/docs, and split the train/dev/test set
    sorted_scores = read_sorted_scores()
    train, dev, test, all = parse_split_data(sorted_scores, train_percent, dev_percent)

    train_pairs = build_pairs(train)
    dev_pairs = build_pairs(dev)
    test_pairs = build_pairs(test)
    print(len(train_pairs), len(dev_pairs), len(test_pairs))

    # read bert vectors
    with open('data/doc_summ_bert_vectors.pkl','rb') as ff:
        all_vec_dic = pickle.load(ff)

    pcc_list = []
    weights_list = []
    for ii in range(epoch_num):
        print('\n=====EPOCH {}====='.format(ii))
        loss = pair_train_rewarder(all_vec_dic, train_pairs, deep_model, optimiser, batch_size, device)
        print('--> loss', loss)

        results = test_rewarder(all_vec_dic, dev, deep_model, device)
        for metric in results:
            print('{}\t{}'.format(metric,np.mean(results[metric])))
        pcc_list.append(np.mean(results['pcc']))
        weights_list.append(copy.deepcopy(deep_model.state_dict()))

    idx = np.argmax(pcc_list)
    best_result = pcc_list[idx]
    print('\n======Best results come from epoch no. {}====='.format(idx))

    deep_model.load_state_dict(weights_list[idx])
    test_results = test_rewarder(all_vec_dic, test, deep_model, device)
    print('Its performance on the test set is:')
    for metric in test_results:
        print('{}\t{}'.format(metric,np.mean(test_results[metric])))
    model_weight_name = 'pcc{0:.4f}_'.format(np.mean(test_results['pcc']))
    model_weight_name += 'epoch{}_batch{}_{}_trainPercent{}_lrate{}_{}.model'.format(
        epoch_num, batch_size, train_type, train_percent, learn_rate, model_type
    )

    torch.save(weights_list[idx], os.path.join(MODEL_WEIGHT_DIR, model_weight_name))
    print('\nbest model weight saved to: {}'.format(os.path.join(MODEL_WEIGHT_DIR, model_weight_name)))



