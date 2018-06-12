#
#
#
#
#
#
##################################################################
import os
import argparse
import wget
import tempfile
import tarfile
import numpy as np
import networkx as nx
from ase.io.extxyz import read_xyz
from ase.units import Hartree, eV, Bohr, Ang
from ase.neighborlist import NeighborList

from utils.logger import log
from xyz_parser import xyz_graph_decoder

def download_qm9(url, file):
  if os.path.exists(file):
    log.infov("Found existing QM9 dataset at {}, SKIP downloading!".format(file))
    return
  wget.download(url, out=file)

def parse_xyz(tmp_dir):#, dbpath):
  prop_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo',
                'gap', 'r2', 'zpve', 'energy_U0', 'energy_U', 'enthalpy_H',
                'free_G', 'Cv']
  conversions = [1., 1., 1., 1., Bohr ** 3 / Ang ** 3,
                 Hartree / eV, Hartree / eV, Hartree / eV,
                 Bohr ** 2 / Ang ** 2, Hartree / eV,
                 Hartree / eV, Hartree / eV, Hartree / eV,
                 Hartree / eV, 1.]

  for i, xyzfile in enumerate(os.listdir(tmp_dir)):
    xyzfile = os.path.join(tmp_dir, xyzfile)

    if i % 10000 == 0:
      log.info(str(i) + "/133885 parsed.")
    #if i == 500:
    #  break
    properties = {}
    tmp = os.path.join(tmp_dir, 'tmp.xyz')
    with open(xyzfile, 'r') as f:
      lines = f.readlines()
      l = lines[1].split()[2:]
      for pn, p, c in zip(prop_names, l, conversions):
        properties[pn] = float(p) * c
      with open(tmp, 'wt') as fout:
        for line in lines:
          fout.write(line.replace('*^', 'e'))

    with open(tmp, 'r') as f:
      atoms = list(read_xyz(f, 0))[0]
    
    idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = collect_neighbors(atoms, 20.)

    data = {'_idx_ik': idx_ik, '_idx_jk': idx_jk, '_idx_j': idx_j,
            '_seg_i': seg_i, '_seg_j': seg_j, '_offset': offset,
            '_ratio_j': ratio_j}


def load_qm9(data_dir):
  log.info("Downloading GDB-9 datasets...")
  url = 'https://ndownloader.figshare.com/files/3195389'
  data_dir = os.path.join(data_dir, 'qm9')
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  raw_file = os.path.join(data_dir, 'dsgdb9nsd.xyz.tar.bz2')
  download_qm9(url, raw_file)

  #temp = tempfile.mkdtemp('dsgdb9nsd')
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

  log.info("Parsing XYZ files...")
  #parse_xyz(temp)#, dbpath)
  for i, xyzfile in enumerate(os.listdir(temp)):
    xyzfile = os.path.join(temp, xyzfile)
    g, l = xyz_graph_decoder(xyzfile)




  log.info("All XYZ files parsed.")
  #os.removedirs(temp)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--path', help='Path to QM9 directory')
  args = parser.parse_args()
  if args.path is None:
    args.path = './'

  load_qm9(args.path)
