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
      epoch=10,
      #TODO: more hps

    )


  def __init__(self, params):
    super(MPNN).__init__(params)
  


  def _fprop(self, *args):
    pass







