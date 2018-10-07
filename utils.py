from __future__ import absolute_import
from __future__ import print_function

import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from colorlog import ColoredFormatter

format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
data_format ='%Y-%m-%d %H:%M:%S'
cformat = '%(log_color)s' + format_str
colors = {'DEBUG': 'reset',
          'INFO': 'reset',
          'INFOV': 'bold_cyan',
          'WARNING': 'bold_yellow',
          'ERROR': 'bold_red',
          'CRITICAL': 'bold_red'}
formatter = ColoredFormatter(cformat, data_format, log_colors=colors)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(stream_handler)

INFO_LEVELV_NUM = logging.INFO + 1
logging.addLevelName(INFO_LEVELV_NUM, 'INFOV')
def _infov(self, msg, *args, **kwargs):
  if self.isEnabledFor(INFO_LEVELV_NUM):
    self._log(INFO_LEVELV_NUM, msg, args, **kwargs)
logging.Logger.infov = _infov

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--datapath', default='./dataset/qm9.tfrecords', help='Dataset path')
  parser.add_argument('--batch-size', default=10, help='Batch size (default 10)')
  args = parser.parse_args()

  return args

def plot_roc(y_score, y):
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  plt.figure()
  lw = 2
  plt.plot(fpr[1], tpr[1], color='g', lw=lw, label="1T ROC curve (area = %0.2f)" % roc_auc[1])
  plt.plot(fpr[2], tpr[2], color='r', lw=lw, label="2H ROC curve (area = %0.2f)" % roc_auc[2])
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.title("TITLE roc")
  plt.legend(loc='lower right')
  plt.savefig('./roc.png')
