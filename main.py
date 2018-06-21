"""
Train a neural network model on QM9 dataset.


"""

import os
import argparse

import tensorflow as tf
import numpy as np

# Local modules
from utils.logger import log
#from dataset import xyz_parser

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--datapath', default='./dataset/qm9.bin', help='Path to qm9.bin')
parser.add_argument('--batch-size', default=100, help='Input batch size for training (default 100)')


def main():


  args = parser.parse_args()

  datapath = args.datapath


  with tf.Session() as sess:
    
    sess.run(file)












if __name__ == '__main__':
  main()
