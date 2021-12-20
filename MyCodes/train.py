import os
import time
import torch
import helper.logger as logger
from data_modules.vocab import Vocab
from data_modules.data_loader import data_loaders
from models.model import HiAGM
from helper.configure import Configure
from helper.logger import Logger
from train_modules.criterions import ClassificationLoss
from train_modules. trainer import Trainer

def set_optimizer(config, model):
    params = model.optimize_params_dict()
    if config.dict['train']['optimizer']['type'] == 'Adam':
        return torch.optim.Adam(lr=config.dict['train']['optimizer']['learning_rate'],
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")

def train(config):
    corpus_vocab = Vocab(config, max_size=50000)

    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab)

    hiagm = HiAGM(config, corpus_vocab, model_type="HIAGM_TP", model_mode="TRAIN")
    hiagm.to(config.dict['train']['device_setting']['device'])

    criterion = ClassificationLoss(os.path.join(config.dict['data']['data_dir'], config.dict['data']['hierarchy']),
                                   corpus_vocab.v2i['label'],
                                   recursive_penalty=config.dict['train']['loss']['recursive_regularization']['penalty'],
                                   recursive_constraint=config.dict['train']['loss']['recursive_regularization']['flag'])
    optimize = set_optimizer(config, hiagm)

    trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config)

    for epoch in range(config.dict['train']['start_epoch'], config.dict['train']['end_epoch']):
        trainer.train(train_loader,
                      epoch)

if __name__ == "__main__":
    config = Configure("./config/tree-rcv1-v2.json")

    if config.dict['train']['device_setting']['device'] == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(config.dict['train']['device_setting']['visible_device_list']))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    logger.Logger(config)

    if not os.path.isdir(config.dict['train']['checkpoint']['dir']):
        os.mkdir(config.dict['train']['checkpoint']['dir'])

    train(config)