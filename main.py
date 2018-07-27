"""
Train a neural network model on QM9 dataset.


"""

import os
import tensorflow as tf
import numpy as np

# Local modules
from dataset.data_loader import DataLoader
from utils.logger import log
from utils.config import get_args
from model.mpnn import MPNN
from trainer import Trainer
#from dataset import xyz_parser



def main():

  try:
    args = get_args()

  except:
    log.error('Missing or invalid arguments.')
    exit(0)

  # Initialize hyper parameters
  hparams = MPNN.default_hparams()
  hparams.batch_size = 10
  #hparams.padded_num_nodes = 
  #hparams.node_dim = 
  #hparams.prop_step = 
  hparams.reuse_graph_tensor = True

  # Create model
  graph = tf.Graph()
  with graph.as_default():
    data = DataLoader(args.datapath, hparams)
    model = MPNN(hparams)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Trainer
  trainer = Trainer(model, data, graph, hparams, config)
  
  trainer.train()


if __name__ == '__main__':
  main()
