import logging
import os
import json
from typing import Container
import tqdm
import helper.logger as logger
from collections import Counter

class Vocab(object):
    def __init__(self, config, special_token=['<PADDING>', '<OOV>'], max_size=None):
        super(Vocab, self).__init__()
        logger.info("Building Vocabulary ...")
        self.config = config
        self.corpus_files = {"TRAIN": os.path.join(config.dict['data']['data_dir'], config.dict['data']['train_file']),
                             "VAL": os.path.join(config.dict['data']['data_dir'], config.dict['data']['val_file']),
                             "TEST": os.path.join(config.dict['data']['data_dir'], config.dict['data']['test_file'])}

        self.freqs = {"token": dict(), "label": dict()}
        self.i2v = {"token": dict(), "label": dict()}
        self.v2i = {"token": dict(), "label": dict()}

        if not os.path.isdir(self.config.dict['vocabulary']['dir']):
            os.system("mkdir " + self.config.dict['vocabulary']['dir'])

        token_dir = os.path.join(config.dict['vocabulary']['dir'], config.dict['vocabulary']['vocab_dict'])
        label_dir = os.path.join(config.dict['vocabulary']['dir'], config.dict['vocabulary']['label_dict'])
        vocab_dir = {"token": token_dir, "label": label_dir}

        if os.path.isfile(token_dir) and os.path.isfile(label_dir):
            logger.info('Loading Vocabulary from Cached Dictionary...')
            with open(token_dir, "r", encoding="utf-8") as f_token:
                for idx, line in enumerate(f_token):
                    data = line.rstrip('\n').split('\t')
                    assert len(data) == 2
                    vocab = data[0]
                    self.v2i['token'][vocab] = idx
                    self.i2v['token'][idx] = vocab
            with open(label_dir, "r", encoding="utf-8") as f_label:
                for idx, line in enumerate(f_label):
                    data = line.rstrip('\n').split('\t')
                    assert len(data) == 2
                    vocab = data[0]
                    self.v2i['label'][vocab] = idx
                    self.i2v['label'][idx] = vocab
            for vocab in self.v2i.keys():
                logger.info('Vocabulary of ' + vocab + ' ' + str(len(self.v2i[vocab])))

        else:
            logger.info('Generating Vocabulary from Corpus...')
            self._load_pretrained_embedding_vocab()
            self._count_vocab_from_corpus()
            for vocab in self.freqs.keys():
                logger.info('Vocabulary of ' + vocab + ' ' + str(len(self.freqs[vocab])))
            self._shrink_vocab(max_size)
            for token in special_token:
                self.freqs['token'][token] = 1
            
            for key in self.freqs.keys():
                vocab_list = list(self.freqs[key].keys())
                for idx, vocab in enumerate(vocab_list):
                    self.i2v[key][idx] = vocab
                    self.v2i[key][vocab] = idx
                with open(vocab_dir[key], "w", encoding="utf-8") as fout:
                    for vocab in self.v2i[key].keys():
                        fout.write(vocab + '\t' + str(self.freqs[key][vocab]) + '\n')
        self.padding_index = self.v2i['token']['<PADDING>']
        self.oov_index = self.v2i['token']['<OOV>']

    def _load_pretrained_embedding_vocab(self):
        pretrained_file = self.config.dict['embedding']['token']['pretrained_file']
        with open(pretrained_file, "r", encoding="utf-8") as fin:
            for line in tqdm.tqdm(fin):
                vocab = line.rstrip("\n").split(" ")[0]
                self.freqs['token'][vocab.lower()] = 1
    
    def _count_vocab_from_corpus(self):
        for file in self.corpus_files.values():
            mode = "ALL"
            with open(file, "r", encoding="utf-8") as fin:
                logger.info('Loading ' + file + ' subset...')
                for line in fin:
                    data = json.loads(line.rstrip())
                    self._count_vocab_from_sample(data, mode)

    def _count_vocab_from_sample(self, line_dict, mode="ALL"):
        '''
        if mode == "ALL":
            for key in line_dict.keys():
                for vocab in line_dict[key]:
                    self.freqs[key][vocab] += 1
        else:
            for vocab in line_dict['token']:
                self.freqs['token'][vocab] += 1
        '''
        for k in self.freqs.keys():
            if mode == 'ALL':
                for t in line_dict[k]:
                    if t.lower() in self.freqs[k].keys():
                        self.freqs[k][t.lower()] += 1
                    else:
                        self.freqs[k][t.lower()] = 1
            else:
                for t in line_dict['token']:
                    self.freqs['token'][t] += 1

    def _shrink_vocab(self, max_size):
        if max_size is not None:
            tmp_dict = Counter()
            for vocab, num in self.freqs['token'].items():
                tmp_dict[vocab] = num
            tmp_dict_list = tmp_dict.most_common(max_size)
            self.freqs['token'] = Counter()
            for (vocab, num) in tmp_dict_list:
                self.freqs['token'][vocab] = num
        logger.info('Shrinking Vocabulary of tokens: ' + str(len(self.freqs['token'])))