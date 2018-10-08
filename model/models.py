"""
Model class implementations.

"""

import tensorflow as tf


def _fully_connected_var(input_dim, num_layers, fc_dim, output_dim):
  """Define weights of fully connected layers."""
  W = []
  b = []
  layer_dim = input_dim
  for l in range(num_layers):
    W.append(tf.get_variable('W_{}'.format(l), shape=[layer_dim, fc_dim]))
    b.append(tf.get_variable('b_{}'.format(l), shape=[fc_dim]))
    layer_dim = fc_dim
  W.append(tf.get_variable('W_out', shape=[layer_dim, output_dim]))
  b.append(tf.get_variable('b_out', shape=[output_dim]))

  return W, b

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

  def fprop(self, node_state, edge_state, adj_mat, reuse=False):
    """Compute a_v^t from h_v^{t-1}.
    
    Args:
      node_state (tf.float32): [batch_size, num_nodes, node_dim]
      edge_state: placeholder, not used
      adj_mat (tf.int32): [batch_size, num_nodes, num_nodes]
    """

    if not reuse:
      self._compute_parameter_tying(adj_mat)
    
    return self._GGNN(node_state, adj_mat)

  def _compute_parameter_tying(self, adj_mat):
    batch_size = tf.shape(adj_mat)[0]
    num_nodes = tf.shape(adj_mat)[1]
    with tf.name_scope('precompute_graph'):
      a_in_gather = tf.gather(self.matrix_in, adj_mat, name='a_in_gather')
      a_out_gather = tf.gather(self.matrix_out, tf.transpose(adj_mat, [0, 2, 1]), name='a_out_gather')
      a_in = tf.transpose(a_in_gather, [0, 1, 3, 2, 4], name='a_in_tp')
      a_out = tf.transpose(a_out_gather, [0, 1, 3, 2, 4], name='a_out_tp')
      self.a_in = tf.reshape(
        a_in,
        shape=[-1, self.params.node_dim * num_nodes, self.params.node_dim * num_nodes],
        name='a_in')
      #a_in_flat = tf.Print(a_in_flat, [tf.shape(a_in_flat)], '~~~~~~~~~~~~~~~: ')
      self.a_out = tf.reshape(
        a_out,
        shape=[-1, self.params.node_dim * num_nodes, self.params.node_dim * num_nodes],
        name='a_out')

  def _GGNN(self, node_state, adj_mat):
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

    batch_size = tf.shape(node_state)[0]
    num_nodes = tf.shape(node_state)[1]
    with tf.name_scope('GGNN'):
      h_flat = tf.reshape(
        node_state,
        shape=[-1, self.params.node_dim * num_nodes, 1],
        name='h_flat')

      a_in_mult = tf.reshape(
        tf.matmul(self.a_in, h_flat), 
        shape=[batch_size * num_nodes, self.params.node_dim], name='a_in_mult')
      a_out_mult = tf.reshape(
        tf.matmul(self.a_out, h_flat),
        shape=[batch_size * num_nodes, self.params.node_dim], name='a_out_mult')

      a_concat = tf.concat([a_in_mult, a_out_mult], axis=1, name='a_concat')
      a_t = tf.nn.bias_add(a_concat, self.bias, name='a_t_bias')
      message = tf.reshape(a_t, shape=[batch_size, num_nodes, 2 * self.params.node_dim], name='message')

    return message


class EdgeMessagePassing(Model):
  def __init__(self, params):
    super(EdgeMessagePassing, self).__init__(params)
    self.init_graph()

  def _init_graph(self):
    """Edge network"""
    with tf.variable_scope('edge_nn_in'):
      self.W_in, self.b_in = _fully_connected_var(
        self.params.edge_dim,
        self.params.edge_nn_layers,
        self.params.edge_fc_dim,
        self.params.node_dim**2)

    with tf.variable_scope('edge_nn_out'):
      self.W_out, self.b_out = _fully_connected_var(
        self.params.edge_dim,
        self.params.edge_nn_layers,
        self.params.edge_fc_dim,
        self.params.node_dim**2)

  def _compute_parameter_tying(self, edge_state):
    if self.params.activation == 'relu':
      act = tf.nn.relu
    elif self.params.activation == 'tanh':
      act = tf.tanh
    else:
      raise ValueError("Invalid activation: {}".format(self.params.activation))

    batch_size = tf.shape(edge_state)[0]
    num_nodes = tf.shape(edge_state)[1]
    with tf.name_scope('precompute_edge_nn'):
      with tf.name_scope('edge_tying_in'):
        edge_mat_in = tf.reshape(
          edge_state, 
          shape=[batch_size * num_nodes * num_nodes, self.params.edge_dim], 
          name='edge_mat_in')
        for l in range(self.params.edge_nn_layers):
          edge_mat_in = act(tf.matmul(edge_mat_in, self.W_in[l]) + self.b_in[l])
        edge_mat_in = tf.matmul(edge_mat_in, self.W_in[-1]) + self.b_in[-1]
        a_in = tf.reshape(edge_mat_in, shape=[batch_size, num_nodes, num_nodes, self.params.node_dim, self.params.node_dim])
        self.a_in = tf.reshape(
          tf.transpose(a_in, [0, 1, 3, 2, 4]), 
          shape=[batch_size, num_nodes * self.params.node_dim, num_nodes * self.params.node_dim],
          name='a_in')

      with tf.name_scope('edge_tying_out'):
        edge_mat_out = tf.reshape(
          tf.transpose(edge_state, [0, 2, 1, 3]), 
          shape=[batch_size * num_nodes * num_nodes, self.params.edge_dim], 
          name='edge_mat_out')
        for l in range(self.params.edge_nn_layers):
          edge_mat_out = act(tf.matmul(edge_mat_out, self.W_out[l]) + self.b_out[l])
        edge_mat_out = tf.matmul(edge_mat_out, self.W_out[-1]) + self.b_out[-1]
        a_out = tf.reshape(edge_mat_out, shape=[batch_size, num_nodes, num_nodes, self.params.node_dim, self.params.node_dim])
        self.a_out = tf.reshape(
          tf.transpose(a_out, [0, 1, 3, 2, 4]), 
          shape=[batch_size, num_nodes * self.params.node_dim, num_nodes * self.params.node_dim],
          name='a_out')

  def fprop(self, node_state, edge_state, adj_mat, reuse=False):
    """
    Args:
      adj_mat: placeholder, not used
      edge_state (tf.float32): [batch_size, num_nodes, num_nodes, edge_dim]
    """

    if not reuse:
      self._compute_parameter_tying(edge_state)

    return self._EENN(node_state, edge_state) 

  def _EENN(self, node_state, edge_state):
    batch_size = tf.shape(node_state)[0]
    num_nodes = tf.shape(node_state)[1]
    with tf.name_scope('EENN'):
      h_flat = tf.reshape(
        node_state,
        shape=[-1, self.params.node_dim * num_nodes, 1],
        name='h_flat')

      a_in_mult = tf.reshape(
        tf.matmul(self.a_in, h_flat), 
        shape=[batch_size * num_nodes, self.params.node_dim], name='a_in_mult')
      a_out_mult = tf.reshape(
        tf.matmul(self.a_out, h_flat),
        shape=[batch_size * num_nodes, self.params.node_dim], name='a_out_mult')

      a_concat = tf.concat([a_in_mult, a_out_mult], axis=1, name='a_concat')
      #a_t = tf.nn.bias_add(a_concat, self.bias, name='a_t_bias')
      a_t = a_concat
      message = tf.reshape(a_t, shape=[batch_size, num_nodes, 2 * self.params.node_dim], name='message')

    return message


class UpdateFunction(Model):
  def __init__(self, params):
    super(UpdateFunction, self).__init__(params)
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

    self.w_z = tf.get_variable('W_z', shape=[2 * self.params.node_dim, self.params.node_dim])
    self.u_z = tf.get_variable('U_z', shape=[self.params.node_dim, self.params.node_dim])
    self.w_r = tf.get_variable('W_r', shape=[2 * self.params.node_dim, self.params.node_dim])
    self.u_r = tf.get_variable('U_r', shape=[self.params.node_dim, self.params.node_dim])
    self.w = tf.get_variable('W', shape=[2 * self.params.node_dim, self.params.node_dim])
    self.u = tf.get_variable('U', shape=[self.params.node_dim, self.params.node_dim])

  def fprop(self, node_state, message, mask):
    return self._GRU(node_state, message, mask)

  def _GRU(self, node_state, message, mask):
    """Gated Recurrent Units (Cho et al., 2014)
    
    Args:
      node_state (tf.float32): [batch_size, num_nodes, node_dim]
      message (tf.float32): [batch_size, num_nodes, 2 * node_dim]
      mask (tf.bool): [batch_size, num_nodes]

    Return new node_state:
      h_t_rs: [batch_size, num_nodes, node_dim]
    """

    batch_size = tf.shape(node_state)[0]
    num_nodes = tf.shape(node_state)[1]
    with tf.name_scope('GRU'):
      h_rs = tf.reshape(node_state, shape=[batch_size * num_nodes, -1], name='h_rs')
      m_rs = tf.reshape(message, shape=[batch_size * num_nodes, -1], name='m_rs')

      z_t = tf.sigmoid(
        tf.matmul(m_rs, self.w_z) + tf.matmul(h_rs, self.u_z), name='z_t')
      r_t = tf.sigmoid(
        tf.matmul(m_rs, self.w_r) + tf.matmul(h_rs, self.u_r), name='r_t')
      h_tilda = tf.tanh(
        tf.matmul(m_rs, self.w) + tf.matmul(tf.multiply(r_t, h_rs), self.u), name='h_tilda')
      h_t = tf.add(tf.multiply(1 - z_t, h_rs), tf.multiply(z_t, h_tilda), name='h_t')
      mask_col = tf.reshape(mask, 
                            shape=[batch_size * num_nodes, 1], 
                            name='mask_col')
      h_t_mask = tf.multiply(h_t, mask_col, name='h_t_mask')
      h_t_rs = tf.reshape(h_t_mask, shape=[batch_size, num_nodes, -1], name='h_t_rs')

      return h_t_rs


class ReadoutFunction(Model):
  def __init__(self, params):
    super(ReadoutFunction, self).__init__(params)
    self.init_graph()

  def _init_graph(self):
    with tf.variable_scope('fully_connected_i'):
      self.W_i, self.b_i = _fully_connected_var(
        self.params.node_dim * 2,
        self.params.num_layers,
        self.params.fc_dim,
        self.params.output_dim)
    with tf.variable_scope('fully_connected_j'):
      self.W_j, self.b_j = _fully_connected_var(
        self.params.node_dim * 2,
        self.params.num_layers,
        self.params.fc_dim,
        self.params.output_dim)

  def fprop(self, hidden_node, input_node, mask):
    return self._graph_level(hidden_node, input_node, mask)

  def _graph_level(self, hidden_node, input_node, mask):
    """Using the Graph-level output described in GG-NN paper
    
    Args:
      hidden_node: [batch_size, num_nodes, node_dim]
      input_node: [batch_size, num_nodes, node_dim]
      mask (tf.bool): [batch_size, num_nodes]

    Return:
      output: [batch_size, output_dim]
    """

    if self.params.activation == 'relu':
      act = tf.nn.relu
    elif self.params.activation == 'tanh':
      act = tf.tanh
    else:
      raise ValueError("Invalid activation: {}".format(self.params.activation))

    batch_size = tf.shape(input_node)[0]
    num_nodes = tf.shape(input_node)[1]
    with tf.name_scope('feedforward_nn'):
      # The concat vector should have dim of [batch_size*num_nodes, 2*node_dim]
      h_concat = tf.reshape(
        tf.concat([hidden_node, input_node], axis=2, name='concat'),
        shape=[batch_size * num_nodes, -1],
        name='h_concat')

      with tf.name_scope('fc_i'):
        h_x = h_concat
        for l in range(self.params.num_layers):
          h_x = act(tf.matmul(h_x, self.W_i[l]) + self.b_i[l])
        i_out = tf.matmul(h_x, self.W_i[-1]) + self.b_i[-1]

      with tf.name_scope('fc_j'):
        h_x = h_concat
        for l in range(self.params.num_layers):
          h_x = act(tf.matmul(h_x, self.W_j[l]) + self.b_j[l])
        j_out = tf.matmul(h_x, self.W_j[-1]) + self.b_j[-1]

      gated_out = tf.multiply(tf.sigmoid(i_out), j_out)
      mask_col = tf.reshape(mask, 
                            shape=[batch_size * num_nodes, 1], 
                            name='mask_col')
      gated_mask = tf.multiply(gated_out, mask_col, name='gated_mask')
      gated_rs = tf.reshape(
        gated_mask,
        shape=[batch_size, num_nodes, self.params.output_dim],
        name='gated_rs')
      output = tf.reduce_sum(gated_rs, axis=1, name='output')

      return output

