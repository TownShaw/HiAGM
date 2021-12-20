#!/usr
# coding:utf-8

import os
import json

class Configure(object):
    def __init__(self, config_json_file=None):
        if config_json_file != None:
            print(os.listdir())
            assert os.path.isfile(config_json_file)
            with open(config_json_file, "r", encoding="utf-8") as fin:
                self.dict = json.load(fin)