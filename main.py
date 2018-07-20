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
#from dataset import xyz_parser



def main():

  try:
    args = get_args()

  except:
    log.error('Missing or invalid arguments.')
    exit(0)

  # Initialize hyper parameters
  hparams = MPNN.default_hparams()

  # Create data pipeline
  data = DataLoader(args.datapath, hparams)

  model = MPNN(hparams)



  with tf.Session() as sess:

    sess.run(data.iterator.initializer)
    g, h, l = sess.run(data.next_elet)
    log.infov(sess.run(tf.shape(g)))
    log.infov(g)


#    # TODO: need to populate this framework
#    for step in range(1, N_step):
#      try:
#        sess.run(train_op)
#      except tf.error.OutOfRangeError:
#        # Reload the iterator when it reaches the end of the dataset
#        sess.run(iterator.initializer, 
#                 feed_dict={_data: #dataset
#                            _label: #label
#                 })
#        sess.run(train_op)
#      
#      if step % display_step == 0 or step == 1:
#
#        loss, acc = sess.run([loss_op, accuracy])
#        log.info()






if __name__ == '__main__':
  main()
