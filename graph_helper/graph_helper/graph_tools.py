import pandas as pd
import networkx as nx
import collections
from operator import itemgetter

def clean_game_titles(t_names, g_names):
  t_names_inverse = collections.defaultdict(list)
  for k,v in t_names.items():
    t_names_inverse[v].append(k)

  to_merge = {}
  for k,v in t_names_inverse.items():
    if len(v) > 1:
      v = sorted(v)
      to_merge[v[0]] = v

  for k,v in to_merge.items():
    for node_to_merge in v[1:]:
      g_names[k]+=g_names[node_to_merge]
      del t_names[node_to_merge]
      del g_names[node_to_merge]

    g_names[k] = sorted(list(set(g_names[k])))

  return to_merge, t_names, g_names

def clean_game_titles_in_graph(G, to_merge, t_names, g_names):
  for k,v in to_merge.items():
    target_node = None
    source_nodes = []
    for node in v:
      if node in nx.nodes(G):
        if target_node is None:
          target_node = node
        else:
          source_nodes.append(node)
    if target_node is not None and len(source_nodes)>0:
      for source_node in source_nodes:
        G = nx.contracted_nodes(G, target_node, source_node)

        if target_node not in t_names.keys() and source_node in t_names.keys():
          t_names[target_node] = t_names[source_node]
          g_names[target_node] = g_names[source_node]
          del t_names[source_node]
          del g_names[source_node]

  total_weight_dict = {}
  for node in G.nodes():
    total_weight = 0.0
    for edge in G.edges(node,data=True):
      total_weight += edge[2]['weight']
    total_weight_dict[node] = total_weight

  nx.set_node_attributes(G, name='total_weight', values=total_weight_dict)

  return G, list(total_weight_dict.values()), t_names, g_names

def build_G(df, min_edge_weight=None, max_nedges=None, return_total_weights=False):
  G = nx.Graph()
  for index, row in df.iterrows():
    if max_nedges is not None:
      if index > max_nedges:
        break
    n1 = row['n1']
    n2 = row['n2']
    weight = row['records']
    if min_edge_weight is not None:
      if weight < min_edge_weight:
        continue
    if G.has_edge(n1,n2):
      raise ValueError('WARNING With this workflow no edge should be added twice!')
      # G[n1][n2]['weight'] += weight
    else:
      G.add_edge(n1,n2,weight=weight)

  total_weight_dict = {}
  for node in G.nodes():
    total_weight = 0.0
    for edge in G.edges(node,data=True):
      total_weight += edge[2]['weight']
    total_weight_dict[node] = total_weight

  nx.set_node_attributes(G, name='total_weight', values=total_weight_dict)

  if return_total_weights:
    return G, list(total_weight_dict.values())
  else:
    return G

def prune_communities(G, min_size=2):
  G_pruned = G.copy()
  node_communities = [n[1]['community'] for n in G.nodes(data=True)]
  for community in set(node_communities):
    if node_communities.count(community) < min_size:
      node_list = [node[0] for node in filter(lambda x: x[1]['community']==community,G.nodes(data = True))]
      for node in node_list:
        for e in G_pruned.edges(node):
          G_pruned.remove_edge(e[0],e[1])
        G_pruned.remove_node(node)
  return G_pruned


def get_community_info(G, t_names, g_names):
  community_info = {}
  community_genre_comp = {}
  all_used_genres = []

  for community in set([n[1]['community'] for n in G.nodes(data=True)]):
    node_list = [node[0] for node in filter(lambda x: x[1]['community']==community,G.nodes(data = True))]
    community_info[community] = {}
    community_info[community]['Titles'] = []
    community_info[community]['Genres'] = collections.defaultdict(int)
    for node in node_list:
      community_info[community]['Titles'].append(t_names.get(node, 'MISSING'))
      for g in g_names[node]:
        community_info[community]['Genres'][g] += 1

  for k,v in community_info.items():
    community_genre_comp[k] = {}
    for g in v['Genres']:
      all_used_genres.append(g)
      if g in community_genre_comp[k].keys():
        community_genre_comp[k][g] += 1
      else:
        community_genre_comp[k][g] = 1
        
  for k1,v1 in community_genre_comp.items():
    total = sum(v1.values())
    for k2,v2 in v1.items():
      community_genre_comp[k1][k2] = float(v2) / float(total)

  all_used_genres = sorted(list(set(all_used_genres)))

  return community_info, community_genre_comp, all_used_genres

def write_community_info(community_info, m_path, fname='community_titles'):
  with open('{0:s}/{1:s}.txt'.format(m_path, fname), 'w') as f:
    for k,v in community_info.items():
      f.write('Community {0:d}:\n'.format(k))
      f.write('---------------\n')
      for t in sorted(v['Titles']):
        f.write('{0:s}\n'.format(t))
      f.write('\n')
  f.close()

def make_predictions(G, top_k=5):
  predictions = collections.defaultdict(set)
  for node in nx.nodes(G):
    edges = list(G.edges(node, data='weight'))
    edges.sort(key=itemgetter(2),reverse=True)
    for edge in edges[:top_k]:
      predictions[node].add(edge[1])

  return predictions
