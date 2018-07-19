"""Parse input arguments"""

import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--datapath', default='./dataset/qm9.tfrecords', help='Dataset path')
  parser.add_argument('--batch-size', default=10, help='Batch size (default 10)')
  args = parser.parse_args()

  return args