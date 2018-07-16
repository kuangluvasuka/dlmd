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
  
  def fprop(self, *args, **kwargs):
    raise NotImplementedError("Subclass must define fprop() method.")



class MessageFunction(Model):
  """Message function class"""

  def __init__(self, params):
    super(MessageFunction, self).__init__(params)
    self._select_function()
    self.init_graph = tf.make_template(self.__class__.__name__, self._init_graph)
    self.init_graph()
  
  def _init_graph(self):
    """Construct learnable variables matrix_in and matrix out.
    
    This function is called in constructor, so every MessageFunction 
    object will generate it's own learnable variables once instantiated.
    """

    self.matrix_in = tf.get_variable(
      'matrix_weights_in',
      shape=[self.params.num_edge_class, self.params.node_dim, self.params.node_dim])
    self.matrix_out = tf.get_variable(
      'matrix_weight_out',
      shape=[self.params.num_edge_class, self.params.node_dim, self.params.node_dim])
    #tf.Print(self._matrix_out, [self._matrix_out])
    self.bias = tf.get_variable('bias', shape=2 * self.params.node_dim)

    #TODO: Add variables for Non-bounding connection
    if self.params.non_edge:
      pass

  def _select_function(self):
    self._function = {
      'ggnn': self._ggnn

    }.get(self.params.message_function)

  def fprop(
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

  def _ggnn(self, node_state, adj_mat):
    """Gated Graph Neural Network for message passing.
    This function implements the Eq.(2) in Li's paper 
    https://arxiv.org/pdf/1511.05493.pdf.

    Args:
      node_state (tf.float32): [batch_size, num_nodes, node_dim]
      adj_mat (tf.int32): [batch_size, num_nodes, num_nodes]
    
    Return:
      message (tf.float32): the message activation through edges 
                        [batch_size, num_nodes, 2 * node_dim]
    """

    num_nodes = adj_mat.shape[1]
    batch_size = adj_mat.shape[0]

    a_in = tf.gather(self.matrix_in, adj_mat)
    a_out = tf.gather(self.matrix_out, tf.transpose(adj_mat, [0, 2, 1]))
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
    message = tf.reshape(a_concat, shape=[batch_size, num_nodes, 2 * self.params.node_dim])
    message = tf.nn.bias_add(a_v, self.bias, name='bias_add')
    #a_v = tf.Print(a_v, [self._bias.shape], "Shape of message tensor is: ")

    return message


class UpdateFunction(Model):
  def __init__(self, params):
    super(UpdateFunction, self).__init__(params)
    self._select_function()
    self._init_graph()
  
  def _init_graph(self):
    """Construct learnable variables for update function.
    Here, I only implement the GRU-like module which is describe in Li's paper
    
    Variables:
      w_z, u_z for Eq.(3): z_v^{t} = sig(w_z * a_v^{t} + u_z * h_v^{t-1})
      w_r, u_r for Eq.(4): r_v^{t} = sig(w_r * a_v^{t} + u_r * h_v^{t-1})
      w, u for Eq.(5):     h~v^{t} = tanh(w * a_v^{t} + u * (r_v^{t} 0 h_v^{t-1}))

      And Eq.(6):          h_v^{t} = (1 - z_v{t}) 0 h_v^{t-1} + z_v^{t} 0 h~v^{t}
    """

    self.w_z = tf.get_variable('w_z', shape=[2 * self.params.node_dim, self.params.node_dim])
    self.u_z = tf.get_variable('u_z', shape=[self.params.node_dim, self.params.node_dim])
    self.w_r = tf.get_variable('w_r', shape=[2 * self.params.node_dim, self.params.node_dim])
    self.u_r = tf.get_variable('u_r', shape=[self.params.node_dim, self.params.node_dim])
    self.w = tf.get_variable('w', shape=[2 * self.params.node_dim, self.params.node_dim])
    self.u = tf.get_variable('u', shape=[self.params.node_dim, self.params.node_dim])

  def _select_function(self):
    self._function = {
      'GRU':  self._gru

    }.get(self.params.update_function)

  def fprop(self, node_state, message):
    return self._function(node_state, message)

  def _gru(self, node_state, message):
    """Gated Recurrent Units (Cho et al., 2014)
    
    Args:
      node_state:
      message (tf.float32): [batch_size, num_nodes, 2 * node_dim]

    Return new node_state:
      h_t: [batch_size, num_nodes, node_dim]
    """

    z = tf.sigmoid(
      tf.matmul(message, self.w_z) + tf.matmul(node_state, self.u_z), name='z_t')
    r = tf.sigmoid(
      tf.matmul(message, self.w_r) + tf.matmul(node_state, self.u_r), name='r_t')
    h_tilda = tf.tanh(
      tf.matmul(message, self.w) + tf.matmul(tf.multiply(r, node_state), self.u), name='h_tilda_t')
    h_t = tf.add(tf.multiply(1 - z, node_state), tf.multiply(z, h_tilda), name='h_t')

    return h_t


class ReadoutFunction(Model):
  pass



