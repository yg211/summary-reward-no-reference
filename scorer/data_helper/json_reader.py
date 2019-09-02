import codecs
import json
import os
import pickle
import re
from collections import OrderedDict, Counter
import random

from newsroom.build import jsonl
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm import tqdm

from resources import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR, FULL_CDM_RAW_DIR


def read_scores(all=False, remove_null=False, data_dir=RAW_DATA_DIR):
    file_path = os.path.join(data_dir, 'lqual{}.jsonl'.format('_all' if all else ''))

    if not os.path.isfile(file_path):
        raise ValueError("The file {} does not exist!".format(file_path))

    scores = []
    id_set = set()

    with jsonl.open(file_path) as file:
        for sample in file:
            entry = OrderedDict()
            entry['id'] = sample['input']['contents']['id']
            entry['ref'] = sample['input']['contents']['reference']
            entry['sys_name'] = sample['input']['contents']['system']
            entry['sys_summ'] = sample['input']['contents']['text']
            del sample['output']['_responses']
            entry['scores'] = sample['output']

            if remove_null and entry['scores']['overall'] is None:
                continue

            scores.append(entry)
            id_set.add(entry['id'])

    return scores, list(id_set)


def read_processed_scores(all=False, remove_null=False, data_dir=PROCESSED_DATA_DIR):
    file_path = os.path.join(data_dir, 'lqual{}.jsonl'.format('_all' if all else ''))

    if not os.path.isfile(file_path):
        raise ValueError("The file {} does not exist!".format(file_path))

    scores, _ = read_scores(all, remove_null)
    scores_dict = {}

    for score in scores:
        key = "{}-{}".format(score['id'], score['sys_name'])

        if key not in scores_dict:
            scores_dict.update({key: score})

    with jsonl.open(file_path) as file:
        for sample in file:
            key = "{}-{}".format(sample['id'], sample['system'])

            if key in scores_dict:
                del sample['prompts']['overall']['gold']
                del sample['prompts']['overall']['human']

                scores_dict[key]['metrics'] = sample['prompts']['overall']

    del_keys = []

    for key, entry in scores_dict.items():
        if 'metrics' not in entry:
            del_keys.append(key)

    for key in del_keys:
        scores_dict.pop(key)

    id_list = list(set([entry['id'] for entry in scores_dict.values()]))

    return list(scores_dict.values()), id_list


def read_articles(data_dir=RAW_DATA_DIR):
    article_path = os.path.join(data_dir, 'articles.jsonl')

    if not os.path.isfile(article_path):
        raise ValueError("The file {} does not exist!".format(article_path))

    articles = []
    article_id_list = []

    with jsonl.open(article_path) as article_file:
        for sample in article_file:
            entry = OrderedDict()
            entry['id'] = sample['id']
            entry['article'] = sample['text']
            articles.append(entry)

            if entry['id'] not in article_id_list:
                article_id_list.append(entry['id'])

    return articles, article_id_list


def read_references(data_dir=RAW_DATA_DIR):
    ref_path = os.path.join(data_dir, 'lqual_all.jsonl')

    if not os.path.isfile(ref_path):
        raise ValueError("The file {} does not exist!".format(ref_path))

    refs = []
    id_list = []

    with jsonl.open(ref_path) as file:
        for sample in file:
            if sample['id'] in id_list:
                continue

            entry = OrderedDict()
            entry['id'] = sample['id']
            entry['ref'] = sample['reference']

            refs.append(entry)
            id_list.append(entry['id'])

    return refs, id_list


def find_idx_by_id(list_of_dic, id):
    for ii in range(len(list_of_dic)):
        if list_of_dic[ii]['id'] == id:
            return ii
    return -1


def read_article_refs(ids=None, as_dict=False, data_dir=RAW_DATA_DIR):
    refs, ref_ids = read_references()

    article_path = os.path.join(data_dir, 'articles.jsonl')

    if not os.path.isfile(article_path):
        raise ValueError("The file {} does not exist!".format(article_path))

    with jsonl.open(article_path) as file:
        for sample in file:
            if sample['id'] not in ref_ids:
                continue

            idx = find_idx_by_id(refs, sample['id'])

            assert idx != -1
            refs[idx]['article'] = sample['text']

    if ids is not None:
        refs = [r for r in refs if r['id'] in ids]

    if as_dict:
        article_ref_dict = {}

        for r in refs:
            article_ref_dict.update({r['id']: {'article': r['article'], 'ref': r['ref']}})

        return article_ref_dict
    else:
        return refs


def read_sorted_samples(score_key='overall', remove_single_samples=True):
    samples = read_samples()
    samples_dict = {}

    for sample in samples:
        if score_key in sample['scores']:
            score_value = sample['scores'][score_key]
        elif score_key in sample['metrics']:
            score_value = sample['metrics'][score_key]
        elif score_key.startswith('h') and score_key[1:] in sample['heuristics']:
            score_value = sample['heuristics'][score_key]
        else:
            score_value = None

        article_id = sample["id"]

        if score_value is None:
            continue

        if article_id not in samples_dict:
            samples_dict.update({article_id: []})

        sample[score_key] = score_value
        samples_dict[article_id].append(sample)

    # Sort the list of samples per article or remove lists with only one entry
    remove_keys = []
    sample = None

    for article_id, samples in samples_dict.items():
        if len(samples) <= 1 and remove_single_samples:
            remove_keys.append(article_id)
        else:
            samples_dict[article_id] = sorted(samples_dict[article_id], key=lambda s: s[score_key])

    for article_id in remove_keys:
        samples_dict.pop(article_id)

    for article_id, samples in samples_dict.items():
        rank = 0

        for i in range(len(samples)):
            samples[i]['summ_id'] = i

        for i in range(len(samples) - 1):
            # Detect equality and check if left hand summary is the reference summary
            if samples[i][score_key] == samples[i + 1][score_key] and samples[i]['sys_name'].lower() == "reference":
                # In this case, swap the order!
                tmp = samples[i]
                samples[i] = samples[i + 1]
                samples[i + 1] = tmp

            samples[i]['rank'] = rank

            if samples[i][score_key] != samples[i + 1][score_key]:
                rank += 1

        samples[-1]['rank'] = rank

    return samples_dict


def read_sorted_scores():
    sorted_scores_path = os.path.join(DATA_DIR, 'sorted_scores.json')
    return json.load(open(sorted_scores_path, "r"))



def read_samples(data_dir=PROCESSED_DATA_DIR):
    sample_file_path = os.path.join(data_dir, 'samples.jsonl.gz')

    if not os.path.isfile(sample_file_path):
        raise ValueError("The file {} does not exist!".format(sample_file_path))

    with jsonl.open(sample_file_path, gzip=True) as sample_file:
        return sample_file.read()


def read_article_refs_full(genre, split, data_dir=FULL_CDM_RAW_DIR, topics=None):
    articles = OrderedDict()
    refs = OrderedDict()
    path = os.path.join(data_dir, genre, split)

    if topics is None:
        file_names = sorted(os.listdir(path))
    else:
        file_names = topics

    for file_name in file_names:
        if file_name[0] == '.':
            continue  # temp files are omitted

        with codecs.open(os.path.join(path, file_name), 'r', errors='ignore', encoding='utf-8') as file:
            key, _ = os.path.splitext(file_name)
            content = file.read().split('\n\n')
            article = content[1]
            reference = content[2]
            table = content[3].split('\n')

            # replace entity with original names
            for entry in table:
                if len(entry.split(':')) == 1:
                    continue

                article = re.sub('{}\s'.format(entry.split(':')[0].strip()),
                                 '{} '.format(entry.split(':')[1].strip()),
                                 article)
                reference = re.sub('{}\s'.format(entry.split(':')[0]).strip(),
                                   '{} '.format(entry.split(':')[1].strip()),
                                   reference)

            article = re.sub('-', '', article)
            articles[key] = [dd.split('\t')[0].strip() for dd in article.split('\n')]

            reference = re.sub('\*', '', reference)
            reference = re.sub('-', '', reference)

            refs[key] = {'summary': reference.split('\n')}

    return list(articles.keys()), articles, refs,


def create_cnn_dailymail_vocab(lower=False):
    counter = Counter()

    for genre in ["cnn", "dailymail"]:
        for split in ["test", "val", "train"]:
            _, arts, refs = read_article_refs_full(genre, split, data_dir=os.path.normpath(
                "E:/master_thesis/data/cnn_dailymail/raw_anon"))

            for art_sents in tqdm(arts.values(), "Process articles"):
                for sent in art_sents:
                    if lower:
                        sent = sent.lower()

                    counter.update(word_tokenize(re.sub("( '|' )", " ' ", sent)))

            for ref_sents in tqdm(refs.values(), "Process references"):
                for sent in ref_sents:
                    if lower:
                        sent = sent.lower()

                    counter.update(word_tokenize(re.sub("( '|' )", " ' ", sent)))

    counter.update(["."])

    pickle.dump(counter, open("cdm_full{}_counter.p".format("_lower" if lower else ""), "wb"))


if __name__ == '__main__':
    create_cnn_dailymail_vocab()
    exit(0)

    sd = read_sorted_samples('rouge-1')
    exit(0)

    samples = read_samples()
    exit(0)

    scores, id_list = read_processed_scores(remove_null=True)
    exit(0)

    scores, id_list = read_scores()
    article_ref = read_article_refs()

    print('\nscore length: {}'.format(len(scores)))
    print('unique id num in scores: {}'.format(len(id_list)))
    # entry = random.choice(scores)
    # for item in entry:
    #    print('{} : {}'.format(item,entry[item]))

    ref_scores = []
    other_scores = []
    none_scores = 0

    for s in scores:
        overall_score = s["scores"]["overall"]
        id = s["id"]

        if overall_score is None:
            none_scores += 1
            continue

        if s["sys_name"] == "reference":
            ref_scores.append(overall_score)
        else:
            other_scores.append(overall_score)

    print("ref mean ", np.mean(ref_scores))
    print("ref std ", np.std(ref_scores))
    print("other mean ", np.mean(other_scores))
    print("other std ", np.std(other_scores))
    print("min ", np.min(ref_scores + other_scores))
    print("max ", np.max(ref_scores + other_scores))
    print("all mean ", np.mean(ref_scores + other_scores))
    print("none scores ", none_scores)

    exit(0)

    print('\nref length : {}'.format(len(article_ref)))
    entry = random.choice(article_ref)
    for item in entry:
        print('{} : {}'.format(item, entry[item]))

    ### get the avg. number of sentences in refs. and in articles.
    ref_sent_nums = []
    ref_token_nums = []
    art_sent_nums = []
    art_token_nums = []
    for entry in article_ref:
        ref = entry['ref']
        article = entry['article']
        ref_sent_nums.append(len(sent_tokenize(ref)))
        ref_token_nums.append(len(ref.split(' ')))
        art_sent_nums.append(len(sent_tokenize(article)))
        art_token_nums.append(len(article.split(' ')))

    print('\n')
    print('ref sent num: max {}, min {}, mean {}, std {}'.format(
        np.max(ref_sent_nums), np.min(ref_sent_nums), np.mean(ref_sent_nums), np.std(ref_sent_nums)
    ))
    print('ref token num: max {}, min {}, mean {}, std {}'.format(
        np.max(ref_token_nums), np.min(ref_token_nums), np.mean(ref_token_nums), np.std(ref_token_nums)
    ))
    print('article sent num: max {}, min {}, mean {}, std {}'.format(
        np.max(art_sent_nums), np.min(art_sent_nums), np.mean(art_sent_nums), np.std(art_sent_nums)
    ))
    print('article token num: max {}, min {}, mean {}, std {}'.format(
        np.max(art_token_nums), np.min(art_token_nums), np.mean(art_token_nums), np.std(art_token_nums)
    ))
