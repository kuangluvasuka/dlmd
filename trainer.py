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
    self.steps_per_epoch = self.hparams.train_set_num // self.hparams.batch_size

    with self.graph.as_default():
      self._make_train_step()
      # TODO: add resume option
      self._initialize_model()

  def _initialize_model(self):
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)

  def _make_train_step(self):
    """Create train step with optimizer in the graph."""
    adj_mat, node_state, label = self.data.iterator.get_next()

    pred = self.model.fprop(node_state, adj_mat)
    self.loss_op = tf.losses.mean_squared_error(label, pred)
    self.accuracy_op = tf.reduce_mean(tf.abs(pred - label))
    #self.accuracy_op = tf.metrics.mean_absolute_error(label, pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.train_op = optimizer.minimize(self.loss_op)

  def train(self):
    with self.graph.as_default():
      # TODO: add resume option
      # TODO: add exception handler

      train_handle = self.sess.run(self.data.train_iter.string_handle())
      valid_handle = self.sess.run(self.data.valid_iter.string_handle())
      format_str = ('%s: loss: %.5f | acc: %.5f | examples/sec: %.2f')
      for epoch in range(self.hparams.epoch_num):
        self.sess.run(self.data.train_iter.initializer)
        self.sess.run(self.data.valid_iter.initializer)

        log.infov('Epoch %i' % epoch)

        #Train
        train_loss, train_acc, train_speed = self._run_epoch('epoch %i (training)' % epoch, train_handle, True)
        log.infov(format_str % ('Train', train_loss, train_acc, train_speed))

        #Validate
        valid_loss, valid_acc, valid_speed = self._run_epoch('epoch %i (evaluating)' % epoch, valid_handle, False)
        log.infov(format_str % ('Validate', valid_loss, valid_acc, valid_speed))

        # TODO: save logs to file

  def _run_epoch(self, *args, **kwargs):
    """A single training epoch, which loops over the number of mini-batches."""
    raise NotImplementedError('Subclass should implement run_eopch() method.')


class Trainer(BaseTrain):
  def __init__(self, model, data, graph, hparams, config):
    super(Trainer, self).__init__(model, data, graph, hparams, config)

  def _run_epoch(self, epoch_name: str, handle, is_training: bool):

    loss = 0
    accuracy = 0
    start_time = time.time()

    for step in range(self.steps_per_epoch):
      if is_training:
        fetch_list = [self.loss_op, self.accuracy_op, self.train_op]
      else:
        fetch_list = [self.loss_op, self.accuracy_op]

      result = self.sess.run(fetch_list, feed_dict={self.data.handle: handle})
      loss += result[0]
      accuracy += result[1]

      if step % 10 == 0:
        log.info('Running %s, batch %d/%d. Loss: %.4f' % (epoch_name,
                                                           step,
                                                           self.steps_per_epoch,
                                                           result[0]))
    
    instance_per_sec = self.hparams.train_set_num / (time.time() - start_time)
    loss = loss / self.steps_per_epoch
    accuracy = accuracy / self.steps_per_epoch

    return loss, accuracy, instance_per_sec

  