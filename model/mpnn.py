"""
Message Passing Neural Network (MPNN) for quantum molecular simulation training.

"""

import tensorflow as tf

from model import models
from utils.logger import log


class MPNN(models.Model):
  """MPNN model inherited from base Model."""
  
  @staticmethod
  def default_hparams():
    return tf.contrib.training.HParams(
      batch_size=1,
      padded_input_dim=73,
      hidden_node_dim=50,
      hidden_edge_dim=50,
      epoch=10,
      #TODO: more hps
      message_function='mpnn'
      update_function=''
      readout_function=''

    )


  def __init__(self, params):
    super(MPNN).__init__(params)
  


  def _fprop(self, *args):
    pass







