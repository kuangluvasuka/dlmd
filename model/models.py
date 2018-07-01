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
    self.function = None
  
  def _fprop(self, *args, **kwargs):
    raise NotImplementedError("Subclass must define _fprop() method")


class MessageFunction(Model):
  """Message function class"""

  def __init__(self, params):
    super(MessageFunction).__init__(params)
    self._select_function(self.params.message_function)

  def _select_function(self, key, *args):
    
    self.function = {
      'mpnn': self._mpnn

    }.get(key)

  def _mpnn(self, node_state, adj_mat):
    #TODO: Need to figure out the matrix operations 
    #tf.matmul(node_state, adj_mat)


  def _fprop(self):
    return self._mpnn(node_state, adj_mat)






class UpdateFunction(Model):
  pass


class ReadoutFunction(Model):
  pass



