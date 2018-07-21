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
    self.init_graph = tf.make_template(self.__class__.__name__, self._init_graph)
  
  def fprop(self, *args, **kwargs):
    raise NotImplementedError("Subclass must define fprop() method.")

  def _init_graph(self):
    raise NotImplementedError("Subclass must define _init_graph() method.")


class MessageFunction(Model):
  """Message function class"""

  def __init__(self, params):
    super(MessageFunction, self).__init__(params)
    self._select_function()
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
    adj_mat):
    """Compute a_v^t from h_v^{t-1}.
    
    Args:
      node_state (tf.float32): [batch_size, num_nodes, node_dim]
      adj_mat (tf.int32): [batch_size, num_nodes, num_nodes]
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

    a_in = tf.gather(self.matrix_in, adj_mat)
    a_out = tf.gather(self.matrix_out, tf.transpose(adj_mat, [0, 2, 1]))
    a_in = tf.transpose(a_in, [0, 1, 3, 2, 4])
    a_out = tf.transpose(a_out, [0, 1, 3, 2, 4])

    a_in_flat = tf.reshape(
      a_in,
      shape=[-1, self.params.node_dim * self.params.padded_num_nodes, self.params.node_dim * self.params.padded_num_nodes])
    a_out_flat = tf.reshape(
      a_out,
      shape=[-1, self.params.node_dim * self.params.padded_num_nodes, self.params.node_dim * self.params.padded_num_nodes])
    h_flat = tf.reshape(
      node_state,
      shape=[-1, self.params.node_dim * self.params.padded_num_nodes, 1])

    a_in_mult = tf.reshape(
      tf.matmul(a_in_flat, h_flat, name='a_in_mult'), 
      shape=[self.params.batch_size * self.params.padded_num_nodes, self.params.node_dim])
    a_out_mult = tf.reshape(
      tf.matmul(a_out_flat, h_flat, name='a_out_mult'),
      shape=[self.params.batch_size * self.params.padded_num_nodes, self.params.node_dim])

    a_concat = tf.concat([a_in_mult, a_out_mult], axis=1, name='ggnn_concat')
    message = tf.reshape(a_concat, shape=[self.params.batch_size, self.params.padded_num_nodes, 2 * self.params.node_dim])
    message = tf.nn.bias_add(message, self.bias, name='bias_add')
    #a_v = tf.Print(a_v, [self._bias.shape], "Shape of message tensor is: ")

    return message


class UpdateFunction(Model):
  def __init__(self, params):
    super(UpdateFunction, self).__init__(params)
    self._select_function()
    self.init_graph()
  
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
      node_state (tf.float32): [batch_size, num_nodes, node_dim]
      message (tf.float32): [batch_size, num_nodes, 2 * node_dim]

    Return new node_state:
      h_t_rs: [batch_size, num_nodes, node_dim]
    """

    h_rs = tf.reshape(node_state, shape=[self.params.batch_size * self.params.padded_num_nodes, -1], name='gru_h_rs')
    m_rs = tf.reshape(message, shape=[self.params.batch_size * self.params.padded_num_nodes, -1], name='gru_m_rs')

    z = tf.sigmoid(
      tf.matmul(m_rs, self.w_z) + tf.matmul(h_rs, self.u_z), name='gru_z_t')
    r = tf.sigmoid(
      tf.matmul(m_rs, self.w_r) + tf.matmul(h_rs, self.u_r), name='gru_r_t')
    h_tilda = tf.tanh(
      tf.matmul(m_rs, self.w) + tf.matmul(tf.multiply(r, h_rs), self.u), name='gru_h_tilda')
    h_t = tf.add(tf.multiply(1 - z, h_rs), tf.multiply(z, h_tilda), name='gru_h_t')
    h_t_rs = tf.reshape(h_t, shape=[self.params.batch_size, self.params.padded_num_nodes, -1], name='gru_h_t_rs')

    return h_t_rs


class ReadoutFunction(Model):
  def __init__(self, params):
    super(ReadoutFunction, self).__init__(params)
    self._select_function()
    self.init_graph()

  def _init_graph(self):
    self.i = tf.get_variable('i', shape=[2 * self.params.node_dim, self.params.output_dim])
    self.j = tf.get_variable('j', shape=[2 * self.params.node_dim, self.params.output_dim])

  def _select_function(self):
    self._function = {
      'graph_level': self._graph_level

    }.get(self.params.readout_function)
  
  def fprop(self, hidden_node, input_node):
    return self._function(hidden_node, input_node)

  def _graph_level(self, hidden_node, input_node):
    """Using the Graph-level output described in GG-NN paper
    
    Args:
      hidden_node: [batch_size, num_nodes, node_dim]
      input_node: [batch_size, num_nodes, node_dim]

    Return:
      output: [batch_size, output_dim]
    """

    h_x = tf.reshape(
      tf.concat([hidden_node, input_node], 2, name='graph_level_concat'),
      shape=[self.params.batch_size * self.params.padded_num_nodes, -1])
    sigm = tf.sigmoid(tf.matmul(h_x, self.i))
    tanh = tf.tanh(tf.matmul(h_x, self.j))
    act = tf.reshape(
      tf.multiply(sigm, tanh),
      shape=[self.params.batch_size, self.params.padded_num_nodes, -1])
    reduce = tf.reduce_sum(act, axis=1)
    output = tf.tanh(reduce)

    return output

