"""
Message Passing Neural Network (MPNN) for quantum molecular simulation training.

"""

import tensorflow as tf

import models
from utils.logger import log


class MPNN(models.Model):
  """MPNN model inherited from base Model."""
  
  @staticmethod
  def default_hparams():
    """Define hyper parameters.

    - padded_input_dim: to keep the input size constant due to the number of atoms per molecule varies
    - node_dim: dimension of node_state, i.e. h_v
    - edge_dim: dimension of edge_state
    - prop_step: step limit for message propagation
    """

    return tf.contrib.training.HParams(
      batch_size=1,
      padded_num_nodes=73,
      node_dim=50,
      edge_dim=50,
      epoch=10,
      prop_step = 6,
      #TODO: more hps
      num_edge_class=5,
      non_edge=False,
      message_function='ggnn',
      update_function='GRU',
      readout_function=''

    )

  def __init__(self, params):
    super(MPNN, self).__init__(params)
    
    self._m_class = models.MessageFunction
    self._u_class = models.UpdateFunction
    
    self.init_graph()

  def _init_graph(self, reuse_graph_tensor=False):
    if reuse_graph_tensor:
      self.m_function = self._m_class(self.params)
      self.u_function = self._u_class(self.params)
    else:
      self.m_function = [self._m_class(self.params) for _ in range(self.params.prop_step)]
      self.u_function = [self._u_class(self.params) for _ in range(self.params.prop_step)]

  def fprop(self, input_node, adj_mat, reuse_graph_tensor=False):
    """If reuse_graph_tensor is True, create a single m_function for propogation. Otherwise create T m_functions, and call them in turn for T steps.
    """

    h_t = input_node
    for t in range(self.params.prop_step):
      if reuse_graph_tensor:
        message = self.m_function.fprop(h_t, adj_mat)
        h_t = self.u_function.fprop(h_t, message)
      else:
        message = self.m_function[t].fprop(h_t, adj_mat)
        h_t = self.u_function[t].fprop(h_t, message)
    
    message = tf.Print(message, [h_t.shape], 'Returned message is: ')
    return message



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





