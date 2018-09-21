"""
Training and evaluation classes for MPNN.

"""

import tensorflow as tf
import time

from utils.logger import log

class BaseTrain:
  def __init__(self, model, data, graph, hparams, config):
    self.model = model
    self.data = data
    self.hparams = hparams
    self.graph = graph
    self.sess = tf.Session(graph=self.graph, config=config)

    with self.graph.as_default():
      self._make_train_step()
      # TODO: add resume option
      self._initialize_model()

  def _initialize_model(self):
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)
    
    writer = tf.summary.FileWriter('./tf_logs', self.sess.graph)

  def _make_train_step(self):
    """Create train step with optimizer in the graph."""
    raise NotImplementedError('Subclass should implement run_eopch() method.')

  def _evaluate(self, epoch_name: str, handle):
    """Evaluation epoch."""

    loss = 0
    accuracy = 0
    steps = self.hparams.valid_batch_num
    start_time = time.time()

    for step in range(steps):
      fetch_list = [self.loss_op, self.accuracy_op]
      batch_result = self.sess.run(fetch_list, feed_dict={self.data.handle: handle})
      loss += batch_result[0]
      accuracy += batch_result[1]

    instance_per_sec = steps * self.hparams.batch_size / (time.time() - start_time)
    loss = loss / steps
    accuracy = accuracy / steps

    return loss, accuracy, instance_per_sec

  def _run_epoch(self, epoch_name: str, handle):
    """A single training epoch, which loops over the number of mini-batches."""

    loss = 0
    accuracy = 0
    start_time = time.time()

    # TODO: refactor: replace this if-else and for-loop block with while-loop and try-except
    steps = self.hparams.train_batch_num
    for step in range(steps):
      fetch_list = [self.loss_op, self.accuracy_op, self.train_op]

      batch_result = self.sess.run(fetch_list, feed_dict={self.data.handle: handle})
      loss += batch_result[0]
      accuracy += batch_result[1]

      if step % self.hparams.log_step == 0:
        log.info('Running %s, batch %d/%d. Loss: %.4f' % (epoch_name,
                                                          step,
                                                          steps,
                                                          batch_result[0]))
   
    instance_per_sec = steps * self.hparams.batch_size / (time.time() - start_time)
    loss = loss / steps
    accuracy = accuracy / steps

    return loss, accuracy, instance_per_sec


  def train(self):
    with self.graph.as_default():
      # TODO: add resume option
      # TODO: add exception handler

      train_handle = self.sess.run(self.data.train_iterator.string_handle())
      valid_handle = self.sess.run(self.data.valid_iterator.string_handle())
      format_str = ('%s: loss: %.5f | acc: %.5f | examples/sec: %.2f')
      for epoch in range(self.hparams.epoch_num):
        self.sess.run(self.data.train_iterator.initializer)
        self.sess.run(self.data.valid_iterator.initializer)

        log.infov('Epoch %i' % epoch)

        #Train
        train_loss, train_acc, train_speed = self._run_epoch('epoch %i (training)' % epoch, train_handle)
        log.infov(format_str % ('Training', train_loss, train_acc, train_speed))

        #Validate
        valid_loss, valid_acc, valid_speed = self._evaluate('epoch %i (evaluating)' % epoch, valid_handle)
        log.infov(format_str % ('Validation', valid_loss, valid_acc, valid_speed))

        # TODO: save logs to file



class TrainerRegression(BaseTrain):
  def __init__(self, model, data, graph, hparams, config):
    super(TrainerRegression, self).__init__(model, data, graph, hparams, config)

  def _make_train_step(self):
    """Create train step with optimizer in the graph."""
    adj_mat, node_state, edge_state, mask, label = self.data.iterator.get_next()

    pred = self.model.fprop(node_state, edge_state, adj_mat, mask)
    self.loss_op = tf.losses.mean_squared_error(label, pred)
    self.accuracy_op = tf.reduce_mean(tf.abs(pred - label))
    #self.accuracy_op = tf.metrics.mean_absolute_error(label, pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
    self.train_op = optimizer.minimize(self.loss_op)


class TrainerClassification(BaseTrain):
  def __init__(self, model, data, graph, hparams, config):
    super(TrainerClassification, self).__init__(model, data, graph, hparams, config)

  def _make_train_step(self):
    adj_mat, node_state, edge_state, mask, label = self.data.iterator.get_next()
    one_hot_label = tf.one_hot(label, self.hparams.output_dim)
    logit = self.model.fprop(node_state, edge_state, adj_mat, mask)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=one_hot_label)
    self.loss_op = tf.reduce_mean(loss)

    pred = tf.argmax(logit, 1)
    correct = tf.equal(pred, label)
    self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
    self.train_op = optimizer.minimize(self.loss_op)

  