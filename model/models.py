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
    raise NotImplementedError("Subclass must define _fprop() method.")



class MessageFunction(Model):
  """Message function class"""

  def __init__(self, params):
    super(MessageFunction, self).__init__(params)
    self._select_function(self.params.message_function)
    self.init_graph = tf.make_template(self.__class__.__name__, self._init_graph)
    self.init_graph()
  
  def _init_graph(self):
    """Construct learnable variables matrix_in and matrix out.
    
    This function is called in constructor, so every MessageFunction 
    object will generate it's own learnable variables once instantiated.
    """

    self._matrix_in = tf.get_variable(
      'matrix_weights_in',
      shape=[self.params.num_edge_class, self.params.node_dim, self.params.node_dim])
    self._matrix_out = tf.get_variable(
      'matrix_weight_out',
      shape=[self.params.num_edge_class, self.params.node_dim, self.params.node_dim])
    #tf.Print(self._matrix_out, [self._matrix_out])
    self._bias = tf.get_variable('bias', shape=2 * self.params.node_dim)

    #TODO: Add variables for Non-bounding connection
    if self.params.non_edge:
      pass

  def _select_function(self, *args):
    self._function = {
      'mpnn': self._mpnn

    }.get(self.params.message_function)

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

    num_nodes = adj_mat.shape[1]
    batch_size = adj_mat.shape[0]

    #TODO: Need to figure out the matrix operations 
    a_in = tf.gather(self._matrix_in, adj_mat)
    a_out = tf.gather(self._matrix_out, tf.transpose(adj_mat, [0, 2, 1]))
    a_in = tf.transpose(a_in, [0, 1, 3, 2, 4])
    a_out = tf.transpose(a_out, [0, 1, 3, 2, 4])

    a_in_flat = tf.reshape(
      a_in,
      shape=[-1, self.params.node_dim * num_nodes, self.params.node_dim * num_nodes])
    a_out_flat = tf.reshape(
      a_out,
      shape=[-1, self.params.node_dim * num_nodes, self.params.node_dim * num_nodes])
    h_flat = tf.reshape(
      node_state,
      shape=[-1, self.params.node_dim * num_nodes, 1])

    a_in_mult = tf.reshape(
      tf.matmul(a_in_flat, h_flat, name='a_in_mult'), 
      shape=[batch_size * num_nodes, self.params.node_dim])
    a_out_mult = tf.reshape(
      tf.matmul(a_out_flat, h_flat, name='a_out_mult'),
      shape=[batch_size * num_nodes, self.params.node_dim])

    a_concat = tf.concat([a_in_mult, a_out_mult], axis=1, name='concat')
    a_v = tf.reshape(a_concat, shape=[batch_size, num_nodes, 2 * self.params.node_dim])
    a_v = tf.nn.bias_add(a_v, self._bias, name='bias_add')
    #a_v = tf.Print(a_v, [self._bias.shape], "Shape of message tensor is: ")

    return a_v





class UpdateFunction(Model):
  pass


class ReadoutFunction(Model):
  pass



