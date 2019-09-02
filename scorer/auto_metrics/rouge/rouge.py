from __future__  import  print_function
import sys
sys.path.append('../../..')

import tempfile
from os import path,makedirs,getcwd
import time
from subprocess import check_output
import re
import shutil
import random

from resources import BASE_DIR, ROUGE_DIR


class RougeScorer(object):
    def __init__(self, rouge_dir, base_dir):
        self.ROUGE_DIR = rouge_dir
        self.reference_summary_temp_filename = "reference_summary.txt"
        config_file = "config.xml"
        tt = repr(random.randint(0,100000))+repr(time.time())
        self.temp_dir = path.join(base_dir,'rouge_temp_files',tt)
        self.temp_config_file = path.join(self.temp_dir, config_file)

        if not path.exists(self.temp_dir):
            makedirs(self.temp_dir)
        if not path.exists(path.join(self.temp_dir,'models')):
            makedirs(path.join(self.temp_dir,'models'))
        # print("created Rouge instance with tmp: '%s'" % self.temp_dir)
        # print("created Rouge instance with summary_file: '%s'" %  self.reference_summary_temp_filename)
        # print("created Rouge instance with config_file: '%s'" % self.temp_config_file)
        # print("created Rouge instance with ROUGE_DIR: ", self.ROUGE_DIR)

    def create_config(self, peers, models, models_dir):
        config_file = "<EVAL ID=\"1\">\n"
        config_file += "<PEER-ROOT>\n"
        config_file += self.temp_dir + "\n"
        config_file += "</PEER-ROOT>\n"
        config_file += "<MODEL-ROOT>\n"
        config_file += models_dir + "\n"
        config_file += "</MODEL-ROOT>\n"

        config_file += "<INPUT-FORMAT TYPE=\"SPL\">\n</INPUT-FORMAT>\n"
        config_file += "<PEERS>\n"
        for i, peer in enumerate(peers):
            config_file += "<P ID=\"" + str(i + 1) + "\">" + peer + "</P>\n"
        config_file += "</PEERS>\n"

        config_file += "<MODELS>\n"
        config_file += "<M ID=\"" + 'A' + "\">" + 'ref.summary.A'+ "</M>\n"
        #for model, _ in models:
            #model_name = path.basename(model)
            #config_file += "<M ID=\"" + model_name[-1] + "\">" + model_name + "</M>\n"
        config_file += "</MODELS>\n"
        config_file += "</EVAL>\n"

        return config_file

    def extract_results(self, result):
        lines = result.split("\n")
        #print('rouge result lines:\n')
        #for line in lines:
            #print(line)
        result_dict = {}
        prev_exp = ""
        for line in lines:
            x = re.search("([\w\d]+) (ROUGE-[\w\d][\w]?[*]?) Average_(\w): (\d\.\d*) .+", line)
            if x:
                exp_no, rouge_name, stype, score = x.group(1), x.group(2), x.group(3), x.group(4)
                index = exp_no
                rouge_type = rouge_name + " " + stype
                if exp_no != prev_exp:
                    if index not in result_dict:
                        result_dict[index] = {}
                    result_dict[index]["Experiment"] = exp_no
                    prev_exp = exp_no
                result_dict[index][rouge_type] = score
        return result_dict

    def execute_rouge(self):
        cmd = "perl " + self.ROUGE_DIR + "ROUGE-1.5.5.pl -e " + self.ROUGE_DIR + "data " + self.ROUGE_ARGS + ' -a ' + self.temp_config_file
        return check_output(cmd, shell=True)

    def clean(self):
        shutil.rmtree(self.temp_dir)

    def get_scores(self, summary, models):
        with open(path.join(self.temp_dir, self.reference_summary_temp_filename),'w') as ff:
            ff.write(summary)

        models_dir = path.join(self.temp_dir,'models')
        config = self.create_config([self.reference_summary_temp_filename], models, models_dir)

        with open(self.temp_config_file,'w') as ff:
            ff.write(config)

        result = self.execute_rouge()
        result_dict = self.extract_results(result.decode('utf-8'))
        self.clean()

        R1Rscore = float(result_dict["1"]['ROUGE-1 R'])
        R1Fscore = float(result_dict["1"]['ROUGE-1 F'])
        R2Rscore = float(result_dict["1"]['ROUGE-2 R'])
        R2Fscore = float(result_dict["1"]['ROUGE-2 F'])
        RLRscore = float(result_dict["1"]['ROUGE-L R'])
        RLFscore = float(result_dict["1"]['ROUGE-L F'])
        RSURscore = float(result_dict["1"]['ROUGE-SU* R'])
        RSUFscore = float(result_dict["1"]['ROUGE-SU* F'])
        dic = {}
        dic['ROUGE-1-R'] = R1Rscore
        dic['ROUGE-1-F'] = R1Fscore
        dic['ROUGE-2-R'] = R2Rscore
        dic['ROUGE-2-F'] = R2Fscore
        dic['ROUGE-L-R'] = RLRscore
        dic['ROUGE-L-F'] = RLFscore
        dic['ROUGE-SU*-R'] = RSURscore
        dic['ROUGE-SU*-F'] = RSUFscore
        return dic

    def __call__(self, summary, models):
        self.ROUGE_ARGS = '-c 95 -2 -1 -U -r 1000 -n 2 -w 1.2 '
        with open(path.join(self.temp_dir,'models','ref.summary.A'),'w') as ff:
            ff.write(models)
        return self.get_scores(summary, models)


if __name__ == '__main__':
    rouge_agent = RougeScorer(ROUGE_DIR, BASE_DIR)
    summary = 'Obama visits Beijing.'
    ref = 'President Obama starts his journey to Beijing, China today. He will talk to the president of China and will visit four cities in China.'

    rouge_dic = rouge_agent(summary,ref)
    for metric in rouge_dic:
        print('{}\t{}'.format(metric,rouge_dic[metric]))