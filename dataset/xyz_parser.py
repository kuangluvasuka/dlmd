#
#
##########################################
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
        g.add_edge(i, j, 
                   b_type=e_ij.GetBondType(), 
                   distance=np.linalg.norm(g.node[i]['coord'] - g.node[j]['coord']))
      else:
        # Unbonded
        g.add_edge(i, j,
                   b_type=None,
                   distance=np.linalg.norm(g.node[i]['coord'] - g.node[j]['coord']))

  return g, l

def qm9_nodes(g, hydrogen=False):
  pass

def qm9_edges(g, e_representation='raw_distance'):
  pass

if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser(description='Read a single XYZ file as input')
  parser.add_argument('--path', '-p', nargs=1, help='Specify the path of XYZ file')

  args = parser.parse_args()
  g, l = xyz_graph_decoder(args.path[0])
  print(type(g))
  print(l)
  