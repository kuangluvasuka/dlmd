"""
    xyz parser for mols dataset.

"""

import os
import networkx as nx
import numpy as np


def xyz_graph_decoder(xyzfile):
  with open(xyzfile, 'r') as f:
    # Number of atoms
    na = int(f.readline())

    label = int(f.readline())
    g = nx.Graph()

    # Atom properties
    atom_property = []


#TODO: create a hash that maps str index (a_prop[-1]) to ordered int


    adj_list = []
    for i in range(na):
      a_prop = f.readline()
      a_prop = a_prop.replace('.*^', 'e')
      a_prop = a_prop.replace('*^', 'e')
      a_prop = a_prop.split()
      atom_property.append(a_prop[:-1])
      adj_list.append(int(a_prop[-1]))

    # Add nodes
    for i in range(na):
      g.add_node(adj_list[i],
                 a_symbol=atom_property[i][0],
                 coord=np.array(atom_property[i][1:4]).astype(np.float),
                 pc=float(atom_property[i][4]))

    # Add edges
    for i in range(na):
      e_prop = f.readline()
      e_prop = e_prop.replace('.*^', 'e')
      e_prop = e_prop.replace('*^', 'e')
      e_prop = e_prop.split()

      atom_i = int(e_prop[1])
      num_neighbor = int(e_prop[2])
      for j in range(num_neighbor):
        atom_j = int(e_prop[2*j+3])
        if atom_j in adj_list:
          g.add_edge(atom_i, atom_j,
            bo=float(e_prop[2*j+4]),
            distance=np.linalg.norm(g.node[atom_i]['coord'] - g.node[atom_j]['coord']))
    
    h = _mol_nodes(g)
    g, e = _mol_edges(g)
    
    return g, h, e, l


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
