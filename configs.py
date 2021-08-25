# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint

project_dir = Path(__file__).resolve().parent
dataset_dir = Path('./dataset/').resolve()
save_dir = Path('./SUM_GAN/')
score_dir = Path('./results/SUM-GAN/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self):
        """Configuration Class: set kwargs as class attributes with setattr"""
        parser = argparse.ArgumentParser()
        
        # Mode
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--verbose', type=str2bool, default='true')

        # Model
        parser.add_argument('--input_size', type=int, default=2048)
        parser.add_argument('--hidden_size', type=int, default=500)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--summary_rate', type=float, default=0.3)

        # Train
        parser.add_argument('--n_epochs', type=int, default=50)
        parser.add_argument('--clip', type=float, default=5.0)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--discriminator_lr', type=float, default=1e-5)
        parser.add_argument('--discriminator_slow_start', type=int, default=15)

        # load epoch
        parser.add_argument('--epoch', type=int, default=2)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str



if __name__ == '__main__':
    config = Config()
    import ipdb
    ipdb.set_trace()
