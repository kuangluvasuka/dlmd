# -*- coding: utf-8 -*-

"""
    download.py: Download QM9 dataset and preprocess it into 'qm9.bin' for training.
    
    Usage:
        $ cd <dir_to_download.py>
        $ python download.py [-p dir]

"""

import os
import argparse
import wget
import tarfile
import glob
import numpy as np
import tensorflow as tf
import pickle

from utils.logger import log
from xyz_parser import xyz_graph_decoder

def download_qm9(url, file):
  if os.path.exists(file):
    log.infov("Found existing QM9 dataset at {}, SKIP downloading!".format(file))
    return
  wget.download(url, out=file)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def load_qm9(data_dir):
  log.info("Downloading GDB-9 datasets...")
  url = 'https://ndownloader.figshare.com/files/3195389'
  data_dir = os.path.join(data_dir, 'qm9')
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  raw_file = os.path.join(data_dir, 'dsgdb9nsd.xyz.tar.bz2')
  download_qm9(url, raw_file)

  temp = os.path.join(data_dir, 'dsgdb9nsd')
  if os.path.exists(temp):
    log.infov("Found existing QM9 xyz files at {}, SKIP Extraction!".format(temp))
  else:
    os.mkdir(temp)
    log.info("Extracting files to {} ...".format(temp))
    tar = tarfile.open(raw_file, 'r:bz2')
    tar.extractall(temp)
    tar.close()
    log.info("Extraction complete.")

  log.info("Parsing XYZ files and streaming parsed features to dataset ..")
  writer = tf.python_io.TFRecordWriter('qm9.tfrecords')
  xyzs = glob.glob(os.path.join(temp, '*.xyz'))
  for i, xyz in enumerate(xyzs):
    if i % 10000 == 0:
      log.info(str(i) + '/133885 saved..')
    g, h, e, l = xyz_graph_decoder(xyz)
    g = np.array(g)
    h = np.array(h)
    l = np.array(l)
    num_nodes = g.shape[0]
    feature = {'label': _bytes_feature(l.tostring()),
               'adjacency': _bytes_feature(g.tostring()),
               'node_state': _bytes_feature(h.tostring()),
               'edge_state': _bytes_feature(e.tostring()),
               'num_nodes': _int64_feature(num_nodes)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
  writer.close()

log.info("Dataset saved in file \'qm9.tfrecords\', DONE!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--path', help='Path to QM9 directory')
  args = parser.parse_args()
  if args.path is None:
    args.path = './'

  load_qm9(args.path)
