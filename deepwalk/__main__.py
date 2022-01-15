#! /usr/bin/env python
# -*- coding: utf-8 -*-
import stellargraph as sg
import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import networkx as nx
import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram
import numpy as np
from six import text_type as unicode
from six import iteritems
from six.moves import range

import matplotlib.pyplot as plt 

import psutil
import pandas as pd
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()

node_class = dict()
def read_and_make_graph_nx_version(args):
  global node_class
  node_class = dict()
  edgelist = list()
  class_num = 1
  class_name_to_num = dict()
  with open(args.input + os.sep + 'cora.content', 'r') as f, open(args.input + os.sep + 'cora.cites','r') as f2:
    for line in f:
      l = line.strip().split()
      class_name = l[-1]
      if class_name not in class_name_to_num:
        class_name_to_num[class_name] = class_num
        class_num += 1
      node_class[l[0]] = class_name #class_name_to_num[class_name]
      for line in f2:
        l = line.strip().split()
        edgelist.append((l[1],l[0]))
  
  nx_G = nx.DiGraph()
  nx_G.add_edges_from(edgelist)
  deepwalk_G = graph.from_networkx(nx_G,undirected=False)
  return nx_G, deepwalk_G

def process(args):
  global node_class
  nx_G, G = read_and_make_graph_nx_version(args)
  print("Number of nodes: {}".format(len(G.nodes())))
  num_walks = len(G.nodes()) * args.number_walks
  print("Number of walks: {}".format(num_walks))
  data_size = num_walks * args.walk_length
  print("Data size (walks*length): {}".format(data_size))
  if data_size < args.max_memory_data_size: #메모리에 data가 올라갈 수 있으면
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))               
    print("Training...")
    model = Word2Vec(walks, vector_size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
    model.wv.save_word2vec_format(args.output)
  else:
    print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
    print("Walking...")
    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                         path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                         num_workers=args.workers)
  
  
  node_classification(model,nx_G)
  tsne_visualization(model)
  
def tsne_visualization(model):
  global node_class
  node_ids = model.wv.index_to_key  # list of node IDs
  node_subjects = pd.Series(node_class)
  node_targets = node_subjects.loc[node_ids]

  transform = TSNE  # PCA
  trans = transform(n_components=3)
  node_embeddings_3d = trans.fit_transform(model.wv.vectors)

  alpha = 0.7
  label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
  node_colours = [label_map[target] for target in node_targets]

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  #plt.axes().set(aspect="equal")
  ax.scatter(
      node_embeddings_3d[:, 0],
      node_embeddings_3d[:, 1],
      node_embeddings_3d[:, 2],
      c=node_colours,
      cmap="jet",
      alpha=alpha,
  )
  plt.title("{} visualization of node embeddings".format(transform.__name__))
  plt.show()
  plt.savefig("visualization.png")

def node_classification(model, nx_G):
  K = 7
  kmeans = KMeans(n_clusters=K, random_state=0)
  kmeans.fit(model.wv.vectors)
  print(model.wv.vectors)
  
  for n, label in zip(model.wv.index_to_key, kmeans.labels_):
    nx_G.nodes[n]['label'] = label

  for n in nx_G.nodes(data=True):
    if 'label' not in n[1].keys():
      n[1]['label'] = 7
  plt.figure(figsize=(12, 6), dpi=600)
  nx.draw_networkx(nx_G, pos=nx.layout.spring_layout(nx_G), 
  				node_color=[[n[1]['label'] for n in nx_G.nodes(data=True)]], 
					cmap=plt.cm.rainbow,
          node_shape='.',
          font_size='2'
					)
 
  plt.axis('off')
  plt.savefig('img.png', bbox_inches='tight', pad_inches=0)

def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args)

if __name__ == "__main__":
  sys.exit(main())
