"""
Message Passing Neural Network (MPNN) for quantum molecular simulation training.

"""

import tensorflow as tf

from model import models
from utils import log


class MPNN(models.Model):
  """MPNN model inherited from base Model."""
  
  @staticmethod
  def default_hparams():
    """Define hyper parameters.

    Model params:
    - padded_input_dim: to keep the input size constant due to the number of atoms per molecule varies
    - node_dim: dimension of node_state, i.e. h_v
    - edge_dim: dimension of edge_state
    - prop_step: step limit for message propagation
    - reuse_graph_tensor: use the same message and update weights in each propagation step
    - output_dim: dimension of the vector valued output, i.e. y_hat
    - num_layers: number of fully connected layers
    - fc_dim: number of hidden weights in fully connected layer

    Training Params:

    """

    return tf.contrib.training.HParams(
      num_nodes=70,
      node_dim=50,
      edge_dim=50,
      edge_nn_layers=3,
      edge_fc_dim=50,
      fc_dim=200,
      num_layers=3,
      prop_step=6,
      reuse_graph_tensor=False,
      output_dim=12,
      num_edge_class=5,
      non_edge=False,
      message_function='ggnn',
      update_function='GRU',
      readout_function='graph_level',
      activation='relu',
      batch_size=10,
      epoch_num=3,
      train_batch_num=1000,
      valid_batch_num=100,
      test_batch_num=100,
      learning_rate=0.001,
      log_step=10)

  def __init__(self, params):
    super(MPNN, self).__init__(params)
    
    if self.params.message_function == 'ggnn':
      self._m_class = models.MessageFunction
    elif self.params.message_function == 'edgenn':
      self._m_class = models.EdgeMessagePassing
    else:
      raise ValueError(
          "Invalid message function: {}".format(self.params.message_function))
    self._u_class = models.UpdateFunction
    self._r_class = models.ReadoutFunction
    
    self.init_graph()

  def _init_graph(self):
    if self.params.reuse_graph_tensor:
      self.m_function = self._m_class(self.params)
      self.u_function = self._u_class(self.params)
    else:
      self.m_function = [self._m_class(self.params) for _ in range(self.params.prop_step)]
      self.u_function = [self._u_class(self.params) for _ in range(self.params.prop_step)]
    
    self.r_function = self._r_class(self.params)

  def fprop(self, node_state, edge_state, adj_mat, mask):
    """If reuse_graph_tensor is True, create a single m_function for propogation. Otherwise create T m_functions, and call them in turn for T steps.
    """

    h_t = [node_state]
    for t in range(self.params.prop_step):
      if self.params.reuse_graph_tensor:
        message = self.m_function.fprop(h_t[-1], edge_state, adj_mat, reuse=(t != 0))
        h_new = self.u_function.fprop(h_t[-1], message, mask)
      else:
        message = self.m_function[t].fprop(h_t[-1], edge_state, adj_mat)
        h_new = self.u_function[t].fprop(h_t[-1], message, mask)
      h_t.append(h_new)
    
    #message = tf.Print(message, [h_t.shape], 'Returned message is: ')
    output = self.r_function.fprop(h_t[-1], node_state, mask)
    return output



if __name__ == '__main__':
  import numpy as np
  num_nodes = 6
  node_dim = 50
  batch_size = 1
  node_input = np.random.rand(batch_size, num_nodes, node_dim).astype('float32')
  adj = np.random.randint(2, size=(batch_size, num_nodes, num_nodes))

  dataset = tf.data.Dataset.from_tensor_slices((node_input, adj))
  iterator = dataset.make_initializable_iterator()
  next_elem = iterator.get_next()

  hparams = MPNN.default_hparams()
  model = MPNN(hparams)

  #model._fprop(node_input, adj)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(iterator.initializer)
  node_input, adj = next_elem
  node_input = tf.reshape(node_input, [batch_size, num_nodes, node_dim])
  adj = tf.reshape(adj, [batch_size, num_nodes, num_nodes])

  m = model.fprop(node_input, adj)
  sess.run(tf.global_variables_initializer())
  sess.run(m)





