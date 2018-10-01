"""
Training and evaluation classes for MPNN.

"""

import tensorflow as tf
import numpy as np
import time

from utils import log

class BaseTrain:
  def __init__(self, model, data, graph, hparams, config):
    self.model = model
    self.data = data
    self.hparams = hparams
    self.graph = graph
    self.sess = tf.Session(graph=self.graph, config=config)
    self._global_step = 0

    with self.graph.as_default():
      self._make_train_step()
      # TODO: add resume option
      self._initialize_model()

  def _initialize_model(self):
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)
    
    self.merged = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter('./tf_logs', self.sess.graph)

  def _make_train_step(self):
    """Create train step with optimizer in the graph."""
    raise NotImplementedError('Subclass should implement make_train_step() method.')

  def metrics(self, logits, labels):
    """Calculate f1_scores for each class"""

    logits = tf.one_hot(tf.argmax(logits, 1), self.hparams.output_dim)
    epsilon = tf.constant(0.00001)
    f1 = []
    for i in range(self.hparams.output_dim):
      pred = logits[:, i]
      actual = labels[:, i]
      # Count true positives, true negatives, false positives and false negatives
      tp = tf.count_nonzero(pred * actual, dtype=tf.float32)
      tn = tf.count_nonzero((pred - 1) * (actual - 1), dtype=tf.float32)
      fp = tf.count_nonzero(pred * (actual - 1), dtype=tf.float32)
      fn = tf.count_nonzero((pred - 1) * actual, dtype=tf.float32)
      # Calculate accuracy, precision, recall and f1_score
      accuracy = (tp + tn) / (tp + fp + fn + tn)
      precision = tp / (tp + fp + epsilon)
      recall = tp / (tp + fn + epsilon)
      fmeasure = (2 * precision * recall) / (precision + recall + epsilon)
      f1.append(fmeasure)

    # for 1T&2H classification only
    tf.summary.scalar('f1_score_disordered', f1[0])
    tf.summary.scalar('f1_score_1T', f1[1])
    tf.summary.scalar('f1_score_2H', f1[2])

    return f1

  def _evaluate(self, epoch_name: str, handle):
    """Evaluation epoch."""

    loss = 0
    accuracy = 0
    f1_scores = []
    format_str = ('%s: loss: %.5f | acc: %.5f | examples/sec: %.2f \n f1 scores: disorderd %0.5f | 1T %0.5f | 2H %0.5f')
    steps = self.hparams.valid_batch_num
    start_time = time.time()

    for step in range(steps):
      fetch_list = [self.loss_op, self.accuracy_op, self.f1_score]
      batch_result = self.sess.run(fetch_list, feed_dict={self.data.handle: handle})
      loss += batch_result[0]
      accuracy += batch_result[1]
      f1_scores.append(batch_result[2])

    instance_per_sec = steps * self.hparams.batch_size / (time.time() - start_time)
    loss = loss / steps
    accuracy = accuracy / steps
    f1_scores = np.sum(f1_scores, axis=0) / steps
    log.infov(format_str % ('Validation', loss, accuracy, instance_per_sec, f1_scores[0], f1_scores[1], f1_scores[2]))

    return loss, accuracy, instance_per_sec, f1_scores

  def _train_epoch(self, epoch_name: str, handle):
    """A single training epoch, which loops over the number of mini-batches."""

    loss = 0
    accuracy = 0
    format_str = ('%s: loss: %.5f | acc: %.5f | examples/sec: %.2f')
    start_time = time.time()

    # TODO: refactor: replace this if-else and for-loop block with while-loop and try-except
    steps = self.hparams.train_batch_num
    for step in range(steps):
      fetch_list = [self.loss_op, self.accuracy_op, self.train_op, self.merged]
      batch_result = self.sess.run(fetch_list, feed_dict={self.data.handle: handle})
      loss += batch_result[0]
      accuracy += batch_result[1]
      self.writer.add_summary(batch_result[3], self._global_step)
      self._global_step += 1

      if step % self.hparams.log_step == 0:
        log.info('Running %s, batch %d/%d. Loss: %.4f' % (epoch_name,
                                                          step,
                                                          steps,
                                                          batch_result[0]))
   
    instance_per_sec = steps * self.hparams.batch_size / (time.time() - start_time)
    loss = loss / steps
    accuracy = accuracy / steps
    log.infov(format_str % ('Training', loss, accuracy, instance_per_sec))

    return loss, accuracy, instance_per_sec


  def train(self):
    with self.graph.as_default():
      # TODO: add resume option
      # TODO: add exception handler
      train_handle = self.sess.run(self.data.train_iterator.string_handle())
      valid_handle = self.sess.run(self.data.valid_iterator.string_handle())
      for epoch in range(self.hparams.epoch_num):
        self.sess.run(self.data.train_iterator.initializer)
        self.sess.run(self.data.valid_iterator.initializer)

        log.infov('Epoch %i' % epoch)
        #Train
        self._train_epoch('epoch %i (training)' % epoch, train_handle)
        #Validate
        self._evaluate('epoch %i (evaluating)' % epoch, valid_handle)

        # TODO: save logs to file


class TrainerRegression(BaseTrain):
  def __init__(self, model, data, graph, hparams, config):
    super(TrainerRegression, self).__init__(model, data, graph, hparams, config)

  def _make_train_step(self):
    """Create train step with optimizer in the graph."""
    adj_mat, node_state, edge_state, mask, label = self.data.iterator.get_next()

    pred = self.model.fprop(node_state, edge_state, adj_mat, mask)
    self.loss_op = tf.losses.mean_squared_error(label, pred)
    tf.summary.scalar('loss', self.loss_op)
    self.accuracy_op = tf.reduce_mean(tf.abs(pred - label))
    tf.summary.scalar('accuracy', self.accuracy_op)
    #self.accuracy_op = tf.metrics.mean_absolute_error(label, pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
    self.train_op = optimizer.minimize(self.loss_op)


class TrainerClassification(BaseTrain):
  def __init__(self, model, data, graph, hparams, config):
    super(TrainerClassification, self).__init__(model, data, graph, hparams, config)

  def _make_train_step(self):
    adj_mat, node_state, edge_state, mask, label = self.data.iterator.get_next()
    label = tf.squeeze(label)
    one_hot_label = tf.one_hot(label, self.hparams.output_dim)
    logit = self.model.fprop(node_state, edge_state, adj_mat, mask)
    self.f1_score = self.metrics(logit, one_hot_label)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=one_hot_label)
    self.loss_op = tf.reduce_mean(loss)
    tf.summary.scalar('loss', self.loss_op)

    correct = tf.equal(tf.argmax(logit, 1), tf.argmax(one_hot_label, 1))
    self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', self.accuracy_op)

    optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
    self.train_op = optimizer.minimize(self.loss_op)

  