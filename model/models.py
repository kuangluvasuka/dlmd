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
  
  def _fprop(self, *args, **kwargs):
    raise NotImplementedError("Subclass must define _fprop() method")


class MessageFunction(Model):
  pass


class UpdateFunction(Model):
  pass


class ReadoutFunction(Model):
  pass



