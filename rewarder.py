from pytorch_transformers import *
import torch
from torch.autograd import Variable
import numpy as np
import os

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


class Rewarder():
    def __init__(self,weight_path,model_type='linear',vec_dim=1024,device='gpu'):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bert_model = BertModel.from_pretrained('bert-large-uncased')
        self.reward_model = build_model(model_type,vec_dim*2) # times 2 because the input to the model is the concatenation of the doc-vec and the summ-vec
        if 'gpu' in device or 'cuda' in device:
            self.gpu= True
            self.reward_model.load_state_dict(torch.load(weight_path))
            self.reward_model.to('cuda')
        else:
            self.gpu = False
            self.reward_model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))

    def __call__(self, doc, summ):
        doc_vec = list(raw_bert_encoder(self.bert_model,self.bert_tokenizer,[doc],gpu=self.gpu))
        summ_vec = list(raw_bert_encoder(self.bert_model,self.bert_tokenizer,[summ],gpu=self.gpu))
        input_vec = Variable(torch.from_numpy(np.array(doc_vec+summ_vec)).float())
        if self.gpu:
            input_vec = input_vec.to('cuda')
        self.reward_model.eval()
        with torch.no_grad():
            pred_score = self.reward_model(input_vec).data.cpu().numpy().reshape(1,-1)[0][0]
        return pred_score


if __name__ == '__main__':
    doc = 'An information campaign urging the public to "get ready for Brexit" has been launched by the government. ' \
          'The campaign began on Sunday with the launch of a website, gov.uk/brexit.' \
          'Billboards and social media adverts will appear in the coming days and TV adverts will air later this month.' \
          'Michael Gove, who is in charge of no-deal plans, said the adverts encourage "shared responsibility" for preparing to leave the EU on 31 October.' \
          'It has been reported that the campaign could cost as much as £100m as ministers seek to inform people what they might need to do, if anything, ahead of the deadline.'

    summ1 = 'Get ready for Brexit advertising campaign launches'
    summ2 = 'Benedict Pringle, author of the politicaladvertising.co.uk blog, said that, if true, the £100m budget would ' \
            'be roughly double what the National Lottery spends on advertising each year.'
    summ3 = 'An image showing one of the campaign\'s billboards was issued by the Cabinet Office ahead of their rollout this week.'
    summ4 = 'A man has died and another is in hospital following a stabbing at a Tube station.'

    rewarder = Rewarder(os.path.join('trained_models','sample.model'),device='cpu')
    reward1 = rewarder(doc,summ1)
    reward2 = rewarder(doc,summ2)
    reward3 = rewarder(doc,summ3)
    reward4 = rewarder(doc,summ4)

    print(reward1, reward2, reward3, reward4)
