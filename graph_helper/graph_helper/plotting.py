# python
import os
import collections
# from itertools import cycle
import logging
logging.basicConfig(level=logging.WARNING)

from ast import literal_eval
#import warnings

#########################################################
## data
import pandas as pd
import numpy as np
import networkx as nx

########################################################
# plotting

import matplotlib as mpl
mpl.use('Agg', warn=False)
mpl.rcParams['font.family'] = ['HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Arial', 'Lucida Grande', 'sans-serif']
mpl.rcParams['axes.labelsize']  = 16
mpl.rcParams['xtick.top']           = True
mpl.rcParams['ytick.right']         = True
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['xtick.labelsize']     = 13
mpl.rcParams['ytick.labelsize']     = 13
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['xtick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['xtick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['xtick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['xtick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['xtick.minor.pad']     = 1.4  # distance to the minor tick label in points
mpl.rcParams['ytick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['ytick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['ytick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['ytick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['ytick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['ytick.minor.pad']     = 1.4  # distance to the minor tick label in points
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from matplotlib import gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.ticker as ticker

from palettable import tableau # https://jiffyclub.github.io/palettable/

# default_colors = tableau.Tableau_20.colors
# default_colors_mpl = cycle(tableau.Tableau_20.mpl_colors)
# default_colors_mpl = cycle(tableau.Tableau_10.mpl_colors)
# default_colors_mpl = tableau.Tableau_20.mpl_colors
default_colors10_mpl = tableau.Tableau_10.mpl_colors
default_colors20_mpl = tableau.Tableau_20.mpl_colors

########################################################
# Set common plot parameters
vsize = 11 # inches
# aspect ratio width / height
aspect_ratio_single = 4.0/3.0
aspect_ratio_multi = 1.0

canvas_width = 600.0

color_max_frac=1.0
color_min_frac=0.15

# Convert between visJS2jupyter and matplotlib symbol names
visJS_to_mpl_symbol={
'dot':'o',
'square':'s',
'triangleDown':'v',
'triangle':'^',
'diamond':'D',
'star':'*'
}

# Define custom color functions for good discrete node colors, min / max edge colors
# It has to be in this string format for visJS2jupyter. See [`return_node_to_color()`](https://github.com/ucsd-ccbb/visJS2jupyter/blob/master/visJS2jupyter/visJS_module.py#L558) and [`return_edge_to_color()`](https://github.com/ucsd-ccbb/visJS2jupyter/blob/master/visJS2jupyter/visJS_module.py#L605) in [visJS_module.py](https://github.com/ucsd-ccbb/visJS2jupyter/blob/master/visJS2jupyter/visJS_module.py) for the normal methods

def my_node_to_color(G,field_to_map='degree'):
  nodes_with_data = [(n[0], max(n[1][field_to_map], 0)) for n in G.nodes(data=True)]
  alpha = 1.0
  nodes,data = zip(*nodes_with_data)
  nodes_with_data = zip(nodes,data)

  node_to_mapField = dict(nodes_with_data)
  color_list = [default_colors10_mpl[node_to_mapField[d]] for d in G.nodes()]

  color_list = [(int(256*c[0]),int(256*c[1]),int(256*c[2]),alpha) for c in color_list]
  node_to_color = dict(zip(list(G.nodes()),['rgba'+str(c) for c in color_list]))

  return node_to_color

def my_edge_to_color_log_transform(weight, weight_max):
  color_to_mult = color_max_frac-color_min_frac
  color_to_add = color_min_frac

  # for cmap, weight scaled to [0, 1], taking the log and acounting for color_max_frac, color_min_frac
  return (np.log(weight)/np.log(weight_max))*color_to_mult + color_to_add

def my_edge_to_color(G,field_to_map='weight'):
  cmap=plt.cm.Greys
  alpha=1.0

  G_edges = G.edges(data=True)
  edges_with_data = [(e[0],e[1],e[2][field_to_map]) for e in G_edges]
  edges1,edges2,data = zip(*edges_with_data)

  # turn off some safety code to make it easier to replicate in mpl, should have safe weight values anyway
  # min_dn0 = min(data)
  # data = [np.log(max(d,min_dn0)) for d in data]  # set the zero d values to minimum non0 value
  # data = [(d-np.min(data)) for d in data] # shift so we don't have any negative values

  data = [np.log(d) for d in data]

  G_edges = G.edges()
  edges_with_data = zip(zip(edges1,edges2),data)

  color_to_mult = 256*(color_max_frac-color_min_frac)
  color_to_add = 256*color_min_frac

  edge_to_mapField = dict(edges_with_data)
  color_list = [np.multiply(cmap(int((float(edge_to_mapField[d])/np.max(list(edge_to_mapField.values())))*color_to_mult+color_to_add)),256) for d in G_edges]

  color_list = [(int(c[0]),int(c[1]),int(c[2]),alpha) for c in color_list]
  edge_to_color = dict(zip(list(G_edges),['rgba'+str(c) for c in color_list]))

  return edge_to_color


# function to convert visJS2jupyter text colors to regular matplotlib colot tuples
def color_str_to_tuple(color_str):
  color_str = color_str.replace('rgba','')
  color_tuple_raw = literal_eval(color_str)
  color_tuple = [x / 256.0 for x in color_tuple_raw[:3]]
  color_tuple.append(color_tuple_raw[3])
  return color_tuple

def plot_graph(G, pos, m_path, edge_bin_size=20, edge_weight_str=None, skip_first_edge_bin=False, fname = 'graph', tag='', inline=False, edge_to_color_all=None, node_to_color=None):
  fig = plt.figure(fname)
  fig.set_size_inches(aspect_ratio_single*vsize, vsize)

  gs = gridspec.GridSpec(1,2, width_ratios=[2.8, 1])
  ax_left = plt.subplot(gs[0])
  ax_left.axis('off')
  ax_left.margins(0.05,0.05)

  # setup edge colors
  color_max_frac=1.0
  color_min_frac=0.15
  if edge_to_color_all is None:
    edge_to_color_all = my_edge_to_color(G)

  if node_to_color is None:
    node_to_color = my_node_to_color(G,field_to_map='community')

  # draw edges by weight, smallest to largest, binned by edge_bin_size. This makes the graph much more readable
  edges=G.edges(data=True)
  edge_weight_class_ordering=sorted(set([np.ceil(edge[2]['weight']/edge_bin_size) for edge in edges]))
  for iweight_class, weight_class in enumerate(edge_weight_class_ordering):
    if skip_first_edge_bin and iweight_class == 0:
      continue
    edges_to_draw=[edge for edge in edges if np.ceil(edge[2]['weight']/edge_bin_size)==weight_class]
    edge_colors = [color_str_to_tuple(edge_to_color_all.get((e[0],e[1]), edge_to_color_all.get((e[1],e[0]), None))) for e in edges_to_draw]
    nx.draw_networkx_edges(G,pos,edgelist=edges_to_draw,ax=ax_left,width=3,edge_color=edge_colors)

  # have to do a call per symbol due to networkx limitations, so we can't really draw in order of weight
  for shape in set((node[1]['symbol'] for node in G.nodes(data = True))):
    node_list = [node[0] for node in filter(lambda x: x[1]['symbol']==shape,G.nodes(data = True))]
    nx.draw_networkx_nodes(G, ax=ax_left, pos=pos, nodelist=node_list,
                           node_color=[color_str_to_tuple(node_to_color[n]) for n in node_list],
                           node_size=40, # constant size
                           # node_size=[20*np.log(G.node[n]['total_weight']) for n in node_list], # TODO
                           node_shape=visJS_to_mpl_symbol[shape]
                          )

  ax_right = plt.subplot(gs[1])
  ax_right.axis('off')
  ax_right.margins(0,0)
  # community legend
  node_communities = [n[1]['community'] for n in G.nodes(data=True)]

  community_leg_objects=[]
  community_colors = {}
  for community in set((node[1]['community'] for node in G.nodes(data = True))):
    node_list = [node[0] for node in filter(lambda x: x[1]['community']==community,G.nodes(data = True))]
    community_leg_objects.append(plt.Line2D([0],[0],marker='s',ls='none',markersize=18,
                                 label='Community {0:d} ({1:d})'.format(community, node_communities.count(community)),
                                 color=color_str_to_tuple(node_to_color[node_list[0]])
                                 ))

  community_leg = plt.legend(community_leg_objects, [ob.get_label() for ob in community_leg_objects], fontsize='large',
                             # bbox_to_anchor=(-0.2,0.86), loc='upper left', borderaxespad=0.0)
                             bbox_to_anchor=(-0.2,0.98), loc='upper left', borderaxespad=0.0)
  community_leg.set_title('Louvain Communities',prop={'size':'large','weight':'bold'})
  community_leg._legend_box.align = 'left'
  community_leg.get_frame().set_edgecolor('white')
  community_leg.get_frame().set_facecolor('white')

  plt.gca().add_artist(community_leg)

  # edge weight colorbar cb1
  edge_weights = [e[2]['weight'] for e in G.edges(data=True)]
  if min(edge_weights) != 1.0:
    print('min edge weight != 1, edge color scale might be off...')
  edge_weight_max = int(max(edge_weights))

  edge_weight_max_limit = 10000 # TODO
  # if abs(float(edge_weight_max-edge_weight_max_limit))/edge_weight_max_limit > 0.02:
  #     print('WARNING edge_weight_max = {0:f} differs from hard coded max of {1:f} by more than 2%!!').format(edge_weight_max, edge_weight_max_limit)

  cb1_tick_labels = [1, 10, 100, 1000, 10000]
  cb1_ticks=[my_edge_to_color_log_transform(l,10000.0) for l in cb1_tick_labels]

  # cax = fig.add_axes([0.718, 0.51, 0.245, 0.045])
  cax = fig.add_axes([0.718, 0.15, 0.245, 0.045])
  cmap = plt.cm.Greys
  norm = mpl.colors.Normalize(vmin=0, vmax=1)
  cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal',ticks=cb1_ticks)
  cb1.set_label('Edge Weight', fontsize='large',weight='bold') #  $W_{ij}$
  cb1.ax.set_xticklabels(cb1_tick_labels)

  if edge_weight_str is not None:
    plt.figtext(0.85, 0.44, edge_weight_str, size=18, ha='center', va='top')

  # plt.figtext(0.85, 0.05, 'Node Size$\propto$ Total Weight', size=18, ha='center', va='top')

#    with warnings.catch_warnings(): # https://stackoverflow.com/questions/22227165/catch-matplotlib-warning
#        warnings.simplefilter("ignore")
#        plt.tight_layout()
  os.makedirs(m_path, exist_ok=True)
  fig.savefig('{0:s}/{1:s}{2:s}.pdf'.format(m_path, fname, tag))
  if inline:
    plt.show()
  plt.close('all')


def plot_community_genre_comp(community_genre_comp, all_used_genres, m_path, fname = 'genre_comps', tag='', inline=False):
  fig = plt.figure(fname)
  fig.set_size_inches(aspect_ratio_single*vsize, vsize)
  gs = gridspec.GridSpec(1,2, width_ratios=[5, 1])
  ax_left = plt.subplot(gs[0])

  x_axis_community = []
  y_axis_by_genre = collections.defaultdict(list)

  for k,v in community_genre_comp.items():
    x_axis_community.append(k)
    for g in all_used_genres:
      y_axis_by_genre[g].append(community_genre_comp[k].get(g, 0.0))

  ind = np.arange(len(x_axis_community))
  bottom = np.zeros(len(x_axis_community))
  ps = []
  for ig,g in enumerate(all_used_genres):
    y_values = np.array(y_axis_by_genre[g])
    p = ax_left.bar(ind, y_values, width=0.4, bottom=bottom, color=default_colors20_mpl[ig % len(default_colors20_mpl)], label=g)
    bottom += y_values
    ps.append(p)

  ax_left.set_xlabel('Community')
  ax_left.set_xticklabels(x_axis_community)
  ax_left.set_xticks(ind)

  ax_left.set_ylabel('Composition')
  ax_left.set_ylim(0., 1.)

  ax_left.xaxis.label.set_size(20)
  ax_left.yaxis.label.set_size(20)

  ax_left.xaxis.set_tick_params(labelsize=15)
  ax_left.yaxis.set_tick_params(labelsize=15)

  ax_right = plt.subplot(gs[1])
  ax_right.axis('off')
  ax_right.margins(0,0)

  leg = ax_right.legend(ps, all_used_genres, bbox_to_anchor=(-0.2,0.98), loc='upper left', borderaxespad=0.0,frameon=False, fontsize="x-large")
  leg._legend_box.align = 'left'
  leg.get_frame().set_edgecolor('none')
  leg.get_frame().set_facecolor('none')

  leg.set_title('Steam Genres',prop={'size':'x-large','weight':'bold'})

  os.makedirs(m_path, exist_ok=True)
  fig.savefig('{0:s}/{1:s}{2:s}.pdf'.format(m_path, fname, tag))
  if inline:
    plt.show()
  plt.close('all')
