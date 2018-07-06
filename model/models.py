"""
Model class implementations.

"""

import tensorflow as tf


class Model(object):
  """Super class for models."""

  def __init__(self, params):
    """Constructor.
       
    Args:
      params - tf.HParams object.
    """

    self.params = params
    self._init_graph()
  
  def _fprop(self, *args, **kwargs):
    raise NotImplementedError("Subclass must define _fprop() method.")

  def _init_graph(self):
    raise NotImplementedError("Subclass must define _init_graph() method.")


class MessageFunction(Model):
  """Message function class"""

  def __init__(self, params):
    super(MessageFunction).__init__(params)
    self._select_function(self.params.message_function)
  
  def _init_graph(self):
    """Construct learnable variables matrix_in and matrix out.
    
    This function is called in base Model's constructor, so every
    MessageFunction object will generate it's own learnable variables
    once instantiated.
    """

    self._matrix_in = tf.get_variable(
      'matrix_weights_in'
      shape=[self.params.num_edge_class, self.params.node_dim, self.params.node_dim])
    self._matrix_out = tf.get_variable(
      'matrix_weight_out',
      shape=[self.params.num_edge_class, self.params.node_dim, self.params.node_dim])
    #tf.Print(self._matrix_out, [self._matrix_out])

    #TODO: Add variables for Non-bounding connection
    if params.non_edge:
      pass

  def _select_function(self, key, *args):
    """
    """
    
    self._function = {
      'mpnn': self._mpnn

    }.get(key)

  def _fprop(
    self, 
    node_state,
    adj_mat,
    reuse_graph_tensors=False):
    """Compute a_v^t from h_v^{t-1}.
    
    Args:
      node_state (tf.float32): [batch_size, num_nodes, node_dim]
      adj_mat (tf.int32): [batch_size, num_nodes, num_nodes]
      resule_graph_tensors (boolean):
    """

    #if not reuse_graph_tensors:
    #  self._init_graph(adj_mat)
    
    #tf.Assert(self._a_in, adj_mat)

    return self._function(node_state, adj_mat)

  def _mpnn(self, node_state, adj_mat):
    """ 
    """

    #TODO: Need to figure out the matrix operations 
    a_in = tf.gather(self._matrix_in, adj_mat)
    a_out = tf.gather(self._matrix_out, tf.transpose(adj_mat, [0, 2, 1]))
    #tf.Print()
    #tf.Assert()

    a_in = tf.matmul(node_state, a_in)
    a_out = tf.matmul(node_state, a_out)
    #tf.Assert()

    #a_in = tf.transpose







class UpdateFunction(Model):
  pass


class ReadoutFunction(Model):
  pass



