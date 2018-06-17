"""
Train a neural network model on QM9 dataset.


"""

import os
import argparse

import tensorflow as tf
import numpy as np

# Local modules
#from dataset import xyz_parser









def main():


  datapath = "./dataset/qm9.bin"

  file = tf.data.Dataset.list_files(datapath)


  with tf.Session() as sess:
    
    sess.run(file)





if __name__ == '__main__':
  main()










