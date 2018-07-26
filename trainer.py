"""
Training and evaluation classes for MPNN.

"""

import tensorflow as tf
import time


class BaseTrain:
  def __init__(self, model, data, hparams):
    self.model = model
    self.data = data
    self.hparams = hparams
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.steps_per_epoch = self.hparams.train_set_num // self.hparams.batch_size

    with self.graph.as_default():
      self._make_train_step()
      # TODO: add resume option
      self._initialize_model()

  def train(self):
    self.sess.run(self.data.iterator.initializer)
    #TODO: a loop calling run epoch
    with self.graph.as_default():
      pass

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

  def _run_epoch(self, *args, **kwargs):
    """A single training epoch, which loops over the number of mini-batches."""
    raise NotImplementedError('Subclass should implement run_eopch() method.')


class Trainer(BaseTrain):
  def __init__(self, sess, model, data, hparams):
    super(Trainer, self).__init__(sess, model, data, hparams)

  def _run_epoch(self, epoch_name: str, is_training: bool):

    loss = 0
    accuracy = 0
    start_time = time.time()

    for step in range(self.steps_per_epoch):
      if is_training:
        fetch_list = [self.loss_op, self.accuracy_op, self.train_op]
      else:
        fetch_list = [self.loss_op, self.accuracy_op]

      batch_loss, batch_acc, _ = self.sess.run(fetch_list)
      loss += batch_loss
      accuracy += batch_acc

      log.info('Running %s, batch %d/%d. Loss: %.4f', % (epoch_name,
                                                         step,
                                                         self.steps_per_epoch,
                                                         batch_loss))
    
    instance_per_sec = self.hparams.train_set_num / (time.time() - start_time)
    loss = loss / self.steps_per_epoch
    accuracy = accuracy / self.steps_per_epoch

    return loss, accuracy, instance_per_sec

  