#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from preprocesing import read_file
from models import bilstm_training,bilstm_testing
from config import Config
from bow_model import *


if __name__ == '__main__':
    arguments = sys.argv[1:]
    
    if len(arguments) < 2:
        print('Please provide train/test -config and the config file path')
        exit(1)
        
    config_path = arguments[2]
    conf = read_file(config_path)
    config = Config(conf)
        
    if arguments[0] == 'train':
        # Training
        if config.model == 'bow':
            # Bow training
            print('\nBow_Training\n------------')
            bow(config,'train')
        if config.model == 'bilstm':
            # BiLSTM training
            print('\nbiLSTM_Training\n------------')
            bilstm_training(config)        
        
    elif arguments[0] == 'test':
        # Testing
        if config.model == 'bow':
            # Bow testing
            print('\nBow_Testing\n------------')
            bow(config,'test')
        if config.model == 'bilstm':
            # BiLSTM training
            print('\nbiLSTM_Testing\n------------')
            bilstm_testing(config)
        
