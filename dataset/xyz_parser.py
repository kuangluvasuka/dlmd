# -*- coding: utf-8 -*-

"""
    xyz_parser.py: Functions to preprocess dataset.

"""

import os
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

def init_graph(prop):
  prop = prop.split()
  g_tag = prop[0]
  g_index = int(prop[1])
  g_A = float(prop[2])
  g_B = float(prop[3]) 
  g_C = float(prop[4]) 
  g_mu = float(prop[5])
  g_alpha = float(prop[6]) 
  g_homo = float(prop[7])
  g_lumo = float(prop[8]) 
  g_gap = float(prop[9])
  g_r2 = float(prop[10])
  g_zpve = float(prop[11]) 
  g_U0 = float(prop[12]) 
  g_U = float(prop[13])
  g_H = float(prop[14])
  g_G = float(prop[15])
  g_Cv = float(prop[16])

  labels = [g_mu, g_alpha, g_homo, g_lumo, g_gap, g_r2, g_zpve, g_U0, g_U, g_H, g_G, g_Cv]
  # Add graph(molecule) attributes
  return nx.Graph(tag=g_tag, index=g_index, A=g_A, B=g_B, C=g_C, mu=g_mu, alpha=g_alpha, homo=g_homo, lumo=g_lumo, gap=g_gap, r2=g_r2, zpve=g_zpve, U0=g_U0, U=g_U, H=g_H, G=g_G, Cv=g_Cv), labels
  
def xyz_graph_decoder(xyzfile):
  with open(xyzfile, 'r') as f:
    # Number of atoms
    na = int(f.readline())

    # Graph properties
    g, l = init_graph(f.readline())

    # Atom properties
    atom_property = []
    for i in range(na):
      a_properties = f.readline()   #lines of Element types, coords, Mulliken partial charges in e
      a_properties = a_properties.replace('.*^', 'e')
      a_properties = a_properties.replace('*^', 'e')
      a_properties = a_properties.split()
      atom_property.append(a_properties)

    # Frequencies
    f.readline()

    # SMILES
    smiles = f.readline()
    smiles = smiles.split()
    smiles = smiles[0]
    
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(m)

  # Create nodes
  for i in range(m.GetNumAtoms()):
    atom_i = m.GetAtomWithIdx(i)

    # Add node(atom) attributes
    g.add_node(i,
               a_type=atom_i.GetSymbol(),
               a_num=atom_i.GetAtomicNum(), 
               acceptor=0, donor=0,
               aromatic=atom_i.GetIsAromatic(), 
               hybridization=atom_i.GetHybridization(),
               num_h=atom_i.GetTotalNumHs(), 
               coord=np.array(atom_property[i][1:4]).astype(np.float),
               pc=float(atom_property[i][4]))
    
  for i in range(len(feats)):
    if feats[i].GetFamily() == 'Donor':
      node_list = feats[i].GetAtomIds()
      for n in node_list:
        g.node[n]['donor'] = 1
    elif feats[i].GetFamily() == 'Acceptor':
      node_list = feats[i].GetAtomIds()
      for n in node_list:
        g.node[i]['acceptor'] = 1
  
  # Create Edges
  for i in range(m.GetNumAtoms()):
    for j in range(m.GetNumAtoms()):
      e_ij = m.GetBondBetweenAtoms(i, j)
      if e_ij is not None:
        # Add edge(bond) attributes
        g.add_edge(i, j, 
                   b_type=e_ij.GetBondType(), 
                   distance=np.linalg.norm(g.node[i]['coord'] - g.node[j]['coord']))
      else:
        # Unbonded
        g.add_edge(i, j,
                   b_type=None,
                   distance=np.linalg.norm(g.node[i]['coord'] - g.node[j]['coord']))
  
  h = _qm9_nodes(g)
  g, e = _qm9_edges(g)

  return g, h, e, l

def _qm9_nodes(g, hydrogen=False):
  """Return node embedding h_v.
  """

  # h is the embedding of atoms in the molecule
  h = []
  for n, d in g.nodes(data=True):
    h_t = []
    # Atom type (One-hot H, C, N, O, F)
    h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
    # Atomic number
    h_t.append(d['a_num'])
    # Partial Charge
    h_t.append(d['pc'])
    # Acceptor
    h_t.append(d['acceptor'])
    # Donor
    h_t.append(d['donor'])
    # Aromatic
    h_t.append(int(d['aromatic']))
    # Hybridization
    h_t += [int(d['hybridization'] == x) for x in [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]]
    # If number hydrogen is enabled
    if hydrogen:
      h_t.append(d['num_h'])
    h.append(h_t)
  return h

def _qm9_edges(g, e_representation='raw_distance'):
  """Return adjacency matrix and distance of edges.
  """
  remove_edges = []
  edge = {}
  for n1, n2, d in g.edges(data=True):
    e_t = []
    # Raw distance function
    if e_representation == 'chem_graph':
      if d['b_type'] is None:
        remove_edges += [(n1, n2)]
      else:
        e_t += [i+1 for i, x in enumerate([Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]) if x == d['b_type']]
    elif e_representation == 'distance_bin':
      if d['b_type'] is None:
        step = (6-2)/8.0
        start = 2
        b = 9
        for i in range(0, 9):
          if d['distance'] < (start+i*step):
            b = i
            break
        e_t.append(b+5)
      else:
        e_t += [i+1 for i, x in enumerate([Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]) if x == d['b_type']]
    elif e_representation == 'raw_distance':
      if d['b_type'] is None:
        remove_edges += [(n1, n2)]
      else:
        e_t.append(d['distance'])
        e_t += [int(d['b_type'] == x) for x in [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]]
    else:
      print('Incorrect Edge representation transform')
      quit()
    if e_t:
      edge[(n1, n2)] = e_t
  for edg in remove_edges:
    g.remove_edge(*edg)

  g = nx.to_numpy_matrix(g)
  e = np.zeros((g.shape[0], g.shape[1], len(list(edge.values())[0])))
  for edg in edge.keys():
    e[edg[0], edg[1], :] = edge[edg]
    e[edg[1], edg[0], :] = edge[edg]
  
  return g, e


##########################################
def mol_graph_decoder(xyzfile):
  with open(xyzfile, 'r') as f:
    # Number of atoms
    na = int(f.readline())

    label = int(f.readline())
    g = nx.Graph()

    # Atom properties
    atom_property = []
    index_hash = {}
    for i in range(na):
      a_prop = f.readline()
      a_prop = a_prop.replace('.*^', 'e')
      a_prop = a_prop.replace('*^', 'e')
      a_prop = a_prop.split()
      atom_property.append(a_prop[:-1])
      index_hash[a_prop[-1]] = i

    # Add nodes
    for i in range(na):
      g.add_node(i,
                 a_symbol=atom_property[i][0],
                 coord=np.array(atom_property[i][1:4]).astype(np.float),
                 pc=float(atom_property[i][4]))

    # Add edges
    for i in range(na):
      e_prop = f.readline()
      e_prop = e_prop.replace('.*^', 'e')
      e_prop = e_prop.replace('*^', 'e')
      e_prop = e_prop.split()

      atom_i = index_hash[e_prop[1]]
      num_neighbor = int(e_prop[2])
      for j in range(num_neighbor):
        try:
          atom_j = index_hash[e_prop[2*j+3]]
          g.add_edge(atom_i, atom_j,
            bo=float(e_prop[2*j+4]),
            distance=np.linalg.norm(g.node[atom_i]['coord'] - g.node[atom_j]['coord']))
        except KeyError:
          continue
    
    h = _mol_nodes(g)
    g, e = _mol_edges(g)

    return g, h, e, label

def _mol_nodes(g):
  h = []
  for n, d in g.nodes(data=True):
    h_t = []
    h_t += [int(d['a_symbol'] == x) for x in ['Mo', 'S']]
    h_t.append(d['pc'])
    h.append(h_t)

  return h

def _mol_edges(g):
  edge = {}
  for n1, n2, d in g.edges(data=True):
    e_t = []
    e_t.append(d['bo'])
    e_t.append(d['distance'])

    edge[(n1, n2)] = e_t

  g = nx.to_numpy_matrix(g)
  e = np.zeros((g.shape[0], g.shape[1], len(list(edge.values())[0])))
  for edg in edge.keys():
    e[edg[0], edg[1], :] = edge[edg]
    e[edg[1], edg[0], :] = edge[edg]

  return g, e


if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser(description='Read a single XYZ file as input')
  parser.add_argument('--path', '-p', nargs=1, help='Specify the path of XYZ file')

  args = parser.parse_args()
  g, h, e, l = xyz_graph_decoder(args.path[0])
  print("Adjacency matrix: \n", g)
  print("Node embedding: \n", h)
  print("Edge: \n", e)
  print("Label: \n", l)
