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
    


  def _select_function(self, key, *args):
    
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
      node_state (tf.float32): [batch_size, num_node, node_dim]
      adj_mat (tf.int32): [batch_size, node_dim, node_dim]
      resule_graph_tensors (boolean):
    """

    #if not reuse_graph_tensors:
    #  self._init_graph(adj_mat)
    
    #tf.Assert(self._a_in, adj_mat)

    return self._function(node_state, adj_mat)

  def _mpnn(self, node_state):
    """ 
    """
    #TODO: Need to figure out the matrix operations 
    a_in = tf.matmul(node_state, self._a_in)
    a_out = tf.matmul(node_state, self._a_out)








class UpdateFunction(Model):
  pass


class ReadoutFunction(Model):
  pass



