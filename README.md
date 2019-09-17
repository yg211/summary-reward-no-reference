# Better Rewards Yield Better Summaries: Learning to Summarise Without References

This project includes the source code accompanying the following paper:

```
@InProceedings{boehm_emnlp2019_summary_reward,
  author    = {Florian B{\" o}hm and Yang Gao and Christian M. Meyer and Ori Shapira and Ido Dagan and Iryna Gurevych},
  title     = {Better Rewards Yield Better Summaries: Learning to Summarise Without References},
  booktitle = {Proceedings of the 2019 Conference on Conference on Empirical Methods in Natural Language Processing {(EMNLP)}},
  month     = November,
  year      = {2019},
  address   = {Hong Kong, China}
}
```

> **Abstract:** Reinforcement Learning (RL) based document summarisation  systems yield state-of-the-art performance in terms of ROUGE scores, because they directly use ROUGE as the rewards during training. However, summaries with high ROUGE scores often receive low human judgement. To find a better reward function that can guide RL to generate human-appealing summaries, we learn a reward function from human ratings on 2,500 summaries. Our reward function only takes the document and system summary as input. Hence, once trained, it can be used to train RL-based  summarisation systems without using any reference summaries. We show that our learned rewards have significantly higher correlation with human ratings than previous approaches. Human evaluation experiments show that, compared to the state-of-the-art supervised-learning systems and ROUGE-as-rewards RL summarisation systems, the RL systems using our learned rewards  during training generate summarieswith higher human ratings.  


Contact person: Yang Gao, yang.gao@rhul.ac.uk

https://sites.google.com/site/yanggaoalex/home

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions

Disclaimer:
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Summary Evaluation Metric Learned from Human Ratings
We learn a summary evaluation function from 2,500 human ratings
on 500 summaries from the CNN/DailyMail dataset. The human ratings
are from [Chaganty et al.'s ACL-2108 work](https://www.aclweb.org/anthology/P18-1060). 
*The learned evaluation function only takes a document and its candidate
summary as input, hence does not require reference summaries.*
This project includes the learned evaluation metric and the code for training it.



## Prerequisties
* Python3 (tested with Python 3.7 on Ubuntu 18.04 LTS)
* Install all packages in requirement.txt.
```bash
pip3 install -r requirements.txt
```

* Download ROUGE package from the [link](https://www.isi.edu/licensed-sw/see/rouge/) and place it in the rouge directory
```bash
mv RELEASE-1.5.5 scorer/auto_metrics/rouge
```

## Use the Learned Evaluation Function
* The pretrained model is at *trained_models/sample.model* 
* An example usage is provided below:
```python
from rewarder import Rewarder
rewarder = Rewarder(os.path.join('trained_models','sample.model'))
article = 'This is an example article. Article includes more information than the summary.'
summary = 'This is an example summary.'
score = rewarder(article,summary)
```        
        
## Measure the Correlation Between Different Metric Scores and the Human Ratings
* *compare_reward.py* is the script for computing the correlation between multiple different metrics and the human ratings. Sample usage:
```bash
python compare_reward.py --metric bert-human --with_ref 1
```

* Metrics supported:
    * ROUGE-1/2-R/F 
    * METEOR
    * BLEU-1/2 
    * InferSent 
    * BERT-based metrics
        * cosine similarity of the vectors generated by the original BERT-Large-Cased model. For texts longer than 512 tokens, we use sliding window;
        * BERT fine-tuned on multiple natural language inference dataset (BERT-NLI), and additionally on the semantic textual similarity datasets (BERT-NLI-STS), both from https://github.com/UKPLab/sentence-transformers); these BERT versions appear after submission of the camera-ready version, hence their performances are not included in the paper;
        * our learned BERT metric.
* For each metric, it can be used in two ways to measure a system summary's quality:
    * with reference: use the metric to compute the similarity score between a system summary and the reference summary.
    * without reference: use the metric to compute the similarity between a system summary and the input document, without using references.
* The correlation between some selected metrics and the human ratings are blow. The full results can be found in our paper (rho: Spearman, prs: Pearson, tau: Kendall).

|                       | rho  | prs  | tau  |
|-----------------------|------|------|------|
| ROUGE-1-F, w/ ref     | .278 | .301 | .237 |
| ROUGE-2-F, w/ ref     | .260 | .277 | .225 |
| METEOR, w/ ref        | .305 | .285 | .266 |
| InferSent, w/ ref     | .311 | .342 | .261 |
|-----------------------|------|------|------|
| BERT-Large, w/ ref    | .298 | .336 | .254 |
| BERT-Large, w/o ref   | .132 | .154 | .113 |
| BERT-NLI, w/ ref      | .309 | .335 | .264 |
| BERT-NLI, w/o ref     | .258 | .313 | .221 |
| BERT-NLI-STS, w/ ref  | .289 | .321 | .248 |
| BERT-NLI-STS, w/o ref | .272 | .321 | .232 |
|-----------------------|------|------|------|
| BERT-MOVER-WMD1, w/ ref | .325 | .308 | .278 |
| BERT-MOVER-WMD2, w/ ref | .323 | .306 | .274 |
| BERT-MOVER-SMD, w/ ref | .331 | .335 | .282 |
|-----------------------|------|------|------|
| Our-Learned, w/ ref   | **.583** | **.609** | **.511** |
| Our-Learned, w/o ref  | **.583** | **.609** | **.511** |
 
## Training the Metric Function 
The training involves two steps: (i) vectorise the documents and summaries, and
(ii) train a linear model on top of the vectors to output scores. We minimise the 
cross-entropy loss during training (see the paper for more details). 
* *Step 1: vectorise documents and summaries.* The code is provided at step1_encode_doc_summ.py. Sample usage: 
```bash
python step1_encode_doc_summ.py
```
        
 We use sliding window to encode texts with more than 512 tokens. The generated vectors are saved as a pickle file at *data/doc_summ_bert_vectors.pkl* .
* *Step 2: training the linear model.* The code is provided at step2_train_rewarder.py. Sample usage:
```bash
python step2_train_rewarder.py --epoch_num 50 --batch_size 32 --train_type pairwise --train_percent 0.64 --dev_percent 0.16 --learn_rate 3e-4 --model_type linear --device gpu
```
 
 The trained model will be saved to the directory *trained_models*. An example model is provided at *trained_models/sample.model*

## License
Apache License Version 2.0

