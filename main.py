"""
Train a neural network model on QM9 dataset.


"""

import os
import tensorflow as tf
import numpy as np

# Local modules
from dataset.data_loader import DataLoader
from utils import log
from utils import get_args
from model.mpnn import MPNN
from trainer import TrainerRegression
from trainer import TrainerClassification
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
  hparams.epoch_num = 100

  hparams.node_dim=10
  hparams.edge_dim=10
  hparams.num_nodes = 40
  hparams.output_dim = 3
  hparams.edge_nn_lay5rs=3
  hparams.edge_fc_dim=50
  hparams.num_layers=3
  hparams.fc_dim=100
  hparams.prop_step=3

  hparams.message_function='edgenn'
  hparams.train_batch_num=5000
  hparams.valid_batch_num=500
  hparams.reuse_graph_tensor=True
  hparams.log_step = 1000

  # Create model
  graph = tf.Graph()
  with graph.as_default():
    data = DataLoader(args.datapath, hparams)
    model = MPNN(hparams)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Trainer
  trainer = TrainerRegression(model, data, graph, hparams, config)
  
  trainer.train()


if __name__ == '__main__':
  main()
