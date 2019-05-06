#!/usr/bin/env python
# coding: utf-8

# # Exploring Steam Genres
# Matthew Epland  
# [phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)

# ### Setup

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx
import community
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import math
from itertools import cycle
from ast import literal_eval
import json
import collections
# import warnings
import math

# For display purposes
get_ipython().magic(u'matplotlib inline')

from visJS2jupyter import visJS_module


# In[2]:


color_order=[0,2,4,6,8,10,14,16,18,12,1,3,5,7,9,11,15,17,19,13]


# In[3]:


output_path = '../output'


# In[4]:


# Define a function to create the output dir, if it already exists don't crash, otherwise raise an exception
# Adapted from A-B-B's response to http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
# Note in python 3.4+ 'os.makedirs(output_path, exist_ok=True)' would handle all of this...
def make_path(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % path)


# ### Define custom color functions for good discrete node colors, min / max edge colors  
# It has to be in this string format for visJS2jupyter. See [`return_node_to_color()`](https://github.com/ucsd-ccbb/visJS2jupyter/blob/master/visJS2jupyter/visJS_module.py#L558) and [`return_edge_to_color()`](https://github.com/ucsd-ccbb/visJS2jupyter/blob/master/visJS2jupyter/visJS_module.py#L605) in [visJS_module.py](https://github.com/ucsd-ccbb/visJS2jupyter/blob/master/visJS2jupyter/visJS_module.py) for the normal methods  

# In[5]:


def my_node_to_color(G,field_to_map='degree'):
    nodes_with_data = [(n[0], max(n[1][field_to_map], 0)) for n in G.nodes(data=True)]

    cmap_nsteps=20
    if len(color_order)!=cmap_nsteps: print "len(color_order)!=cmap_nsteps, you might have problems!!"
    cmap=plt.get_cmap("tab20")
    alpha = 1.0
    
    color_list_raw = cmap(np.linspace(0, 1, cmap_nsteps)) 
  
    nodes,data = zip(*nodes_with_data)
    data = [color_order[d % len(color_order)] for d in data] # TODO not enough colors for communities, needed % len!
    nodes_with_data = zip(nodes,data)

    node_to_mapField = dict(nodes_with_data)
    color_list = [color_list_raw[node_to_mapField[d]] for d in G.nodes()]
    
    color_list = [(int(256*c[0]),int(256*c[1]),int(256*c[2]),alpha) for c in color_list]
    node_to_color = dict(zip(list(G.nodes()),['rgba'+str(c) for c in color_list]))

    return node_to_color


# In[6]:


color_max_frac=1.0
color_min_frac=0.15
    
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


# In[7]:


# Convert between visJS2jupyter and matplotlib symbol names
visJS_to_mpl_symbol={
'dot':'o',
'square':'s',
'triangleDown':'v',
'triangle':'^',
'diamond':'D',
'star':'*'
}


# ## Load Data

# In[8]:


# TODO eventually will need to load all the _i files here and merge them smartly
# for now, just load some of the edges to get started in networkx
edges_path = '../data/edges_0.csv'
nrows=None
df_e_tmp= pd.read_csv(edges_path, dtype={'n1': int, 'n2': int, 'records':int}, nrows=nrows)


# In[9]:


mean_records = df_e_tmp['records'].mean()
print('mean_records = {0:.2f}'.format(mean_records))


# In[10]:


min_edge_weight=15
nedges=2500


# In[11]:


app_genres_path = '../data/app_genres.csv'
app_id_path = '../data/app_id.csv'

df_g = pd.read_csv(app_genres_path, dtype={'appid': int, 'Genre': object})
df_t = pd.read_csv(app_id_path, dtype={'appid': int, 'Title': object})


# In[12]:


g_names = collections.defaultdict(list)
for index, row in df_g.iterrows():
    g_names[row['appid']].append(row['Genre'])


# In[13]:


# autogenerate symbol_list
# g_names = list(df_g['Genre'].unique())
# g_symbols = {k: v for k, v in zip(g_names, cycle(["dot", "diamond", "star", "triangle", "triangleDown", "square"]))}


# In[14]:


if len(df_t[df_t.duplicated(subset=['appid'], keep=False)].index) > 0:
    raise ValueError('Duplicate appid in df_t!')
    
df_t_index = df_t.set_index('appid')
df_t_names = df_t_index[['Title']]
t_names = df_t_names.to_dict()['Title']


# ### Build graph G
# #### Construct G explicitly, edge by edge, so the weights are correct. Annotate nodes with the total weights from all their edges

# In[15]:


def build_G(df, min_edge_weight=None, max_nedges=None):
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

    return G


# In[16]:


df_e = df_e_tmp.sample(frac=1,replace=False,random_state=5).reset_index(drop=True)


# In[17]:


# now actually create graph
G = build_G(df_e, min_edge_weight=min_edge_weight, max_nedges=nedges)


# ### Create positions using the Fruchterman-Reingold force-directed / spring algorithm

# In[18]:


# k=1/sqrt(n) is the default spring / spacing parameter
# Increase max iterations to 100, just in case it needs it
# Need to install networkx from master to access new random_state parameter
# Sets the random seed and allows for reproducable layouts across runs
spring_pos = nx.spring_layout(G,
                              k=4/np.sqrt(nx.number_of_nodes(G)),
                              iterations=100,
                              #random_state=5
                             )


# ### Create communities using the Louvain Method

# In[19]:


# The resolution parameter affects the size of the returned communities
# Best results found with resolution=1, the default unmodified Louvain Method
communities = community.best_partition(G, weight='weight', resolution=1)

nx.set_node_attributes(G, name='community', values=communities)
node_to_color = my_node_to_color(G,field_to_map='community')


# In[20]:


# Setup useful G variables
nodes_all = G.nodes()

# nodes_to_shape = {}
# for n in nodes_all:
    # nodes_to_shape[n] = g_symbols[g_names[n]]
# nx.set_node_attributes(G, name='symbol', values=nodes_to_shape)

edge_to_color_all = my_edge_to_color(G)


# In[21]:


# tmp just make everything a dot
nodes_to_shape = {}
for n in nodes_all:
    nodes_to_shape[n] = 'dot'
nx.set_node_attributes(G, name='symbol', values=nodes_to_shape)


# ## Plot static graphs

# In[22]:


def plot_graph(G, pos, m_path, edge_bin_size=20, edge_weight_str=None, skip_first_edge_bin=False, fname = 'graph', tag='', inline=False):
    fig = plt.figure(fname)
    vsize = 11 # inches
    aspect_ratio_single = 4.0/3.0
    fig.set_size_inches(aspect_ratio_single*vsize, vsize)

    gs = gridspec.GridSpec(1,2, width_ratios=[2.8, 1])
    ax_left = plt.subplot(gs[0])
    ax_left.axis('off')
    ax_left.margins(0.05,0.05)

    # setup edge colors
    color_max_frac=1.0
    color_min_frac=0.15
    
    # draw edges by weight, smallest to largest, binned by edge_bin_size. This makes the graph much more readable
    edges=G.edges(data=True)
    edge_weight_class_ordering=sorted(set([np.ceil(edge[2]['weight']/edge_bin_size) for edge in edges]))
    for iweight_class, weight_class in enumerate(edge_weight_class_ordering):
        if skip_first_edge_bin and iweight_class == 0:
            continue
        edges_to_draw=[edge for edge in edges if np.ceil(edge[2]['weight']/edge_bin_size)==weight_class]
        edge_colors = [color_str_to_tuple(edge_to_color_all.get((e[0],e[1]), edge_to_color_all.get((e[1],e[0]), None))) for e in edges_to_draw]
        nx.draw_networkx_edges(G,pos,edgelist=edges_to_draw,ax=ax_left,width=4,edge_color=edge_colors)

    # have to do a call per symbol due to networkx limitations, so we can't really draw in order of weight
    for shape in set((node[1]['symbol'] for node in G.nodes(data = True))):
        node_list = [node[0] for node in filter(lambda x: x[1]['symbol']==shape,G.nodes(data = True))]
        nx.draw_networkx_nodes(G, ax=ax_left, pos=pos, nodelist=node_list,
                               node_color=[color_str_to_tuple(node_to_color[n]) for n in node_list],
                               node_size=50, # constant size
                               # node_size=[20*np.log(G.node[n]['total_weight']) for n in node_list],
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

    plt.figtext(0.85, 0.05, 'Node Size$\propto$ Total Weight', size=18, ha='center', va='top')

#    with warnings.catch_warnings(): # https://stackoverflow.com/questions/22227165/catch-matplotlib-warning
#        warnings.simplefilter("ignore")
#        plt.tight_layout()
    make_path(m_path)
    fig.savefig('{0:s}/{1:s}{2:s}.pdf'.format(m_path, fname, tag))
    if inline:
        plt.show()
    plt.close('all')


# In[23]:


plot_graph(G, spring_pos, output_path, skip_first_edge_bin=False, fname = 'graph_with_all_communities', inline=True)


# ### Prune small communities

# In[24]:


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


# In[25]:


G_pruned = prune_communities(G, min_size=16)


# In[26]:


plot_graph(G_pruned, spring_pos, output_path, skip_first_edge_bin=False, fname = 'graph_pruned', inline=True)


# ### Describe Communities

# In[27]:


def community_info(G):
    community_info = {}
    for community in set([n[1]['community'] for n in G.nodes(data=True)]):
        node_list = [node[0] for node in filter(lambda x: x[1]['community']==community,G.nodes(data = True))]
        community_info[community] = {}
        community_info[community]['Titles'] = []
        community_info[community]['Genres'] = collections.defaultdict(int)
        for node in node_list:
            community_info[community]['Titles'].append(t_names.get(node, 'MISSING'))
            for g in g_names[node]:
                community_info[community]['Genres'][g] += 1
    return community_info


# In[28]:


community_info = community_info(G_pruned)


# In[29]:


with open('{0:s}/community_titles.txt'.format(output_path), 'w') as f:
    for k,v in community_info.items():
        f.write('Community {0:d}:\n'.format(k))
        f.write('---------------\n')
        for t in v['Titles']:
            f.write('{0:s}\n'.format(t))
        f.write('\n')
    f.close()


# In[30]:


community_genre_comp = {}
all_used_genres = []
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


# In[31]:


def plot_community_genre_comp(community_genre_comp, m_path, fname = 'genre_comps', tag='', inline=False):
    fig = plt.figure(fname)
    vsize = 11 # inches
    aspect_ratio_single = 4.0/3.0
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
        p = ax_left.bar(ind, y_values, width=0.4, bottom=bottom)
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

    leg = ax_right.legend(ps[::-1], all_used_genres,bbox_to_anchor=(-0.2,0.98), loc='upper left', borderaxespad=0.0,frameon=False, fontsize="x-large")
    leg._legend_box.align = 'left'
    leg.get_frame().set_edgecolor('none')
    leg.get_frame().set_facecolor('none')

    leg.set_title('Steam Genres',prop={'size':'x-large','weight':'bold'})

    make_path(m_path)
    fig.savefig('{0:s}/{1:s}{2:s}.pdf'.format(m_path, fname, tag))
    if inline:
        plt.show()
    plt.close('all')


# In[32]:


plot_community_genre_comp(community_genre_comp, output_path, fname = 'genre_comps', inline=True)


# ## Draw interactive graph

# In[33]:


# create nodes_dict
nodes_pruned = G_pruned.nodes(data = False)
seperator = ', '
nodes_dict = [{"id":n,
               "title":("<center><b>{title:s}</b><br>Community {community:d}<br>Node Weight: {total_weight:.0f}<br>Genres: {genres:s}</center>".format(
                   title=t_names.get(n, 'MISSING'),
                   community=communities[n],
                   total_weight=G_pruned.node[n]['total_weight'],
                   genres=seperator.join(g_names[n]),
                   )),
               "x":spring_pos[n][0]*1000,
               "y":(1-spring_pos[n][1])*1000,
               "color":node_to_color[n],
               "node_shape":G_pruned.node[n]['symbol'],
               "node_size":G_pruned.node[n]['total_weight']
              } for n in nodes_pruned]
node_map = dict(zip(nodes_pruned,range(len(nodes_pruned)))) # map to indices for source/target in edges


# In[34]:


# create edges_dict
edges_dict = [{"source":node_map[e[0]],
               "target":node_map[e[1]],
               "title":("<center>n1: {n1:s}<br>n2: {n2:s}<br>Edge Weight: {e_weight:.0f}</center".format(
                   n1=t_names.get(e[0], 'MISSING'),
                   n2=t_names.get(e[1], 'MISSING'),
                   e_weight=e[2]['weight']
               )),
               "color":edge_to_color_all.get((e[0],e[1]), edge_to_color_all.get((e[1],e[0]), None)),
              } for e in G_pruned.edges(data=True)]


# In[35]:


# save the dicts for later viewing
with open('{0:s}/nodes.json'.format(output_path), 'w') as fp:
    json.dump(nodes_dict, fp, sort_keys=True, indent=2)

with open('{0:s}/edges.json'.format(output_path), 'w') as fp:
    json.dump(edges_dict, fp, sort_keys=True, indent=2)


# In[36]:


visJS_module.visjs_network(nodes_dict, edges_dict,
                           node_size_field='node_size', node_size_transform='Math.log', node_size_multiplier=2,
                           node_font_size=0,
                           edge_width=9, edge_title_field="title",
                           physics_enabled=False,
                           graph_title="Interactive Steam Graph",
                           graph_width = 940, graph_height = 600, border_color='black',
                           tooltip_delay = 0, graph_id = 0, config_enabled=False
                          )

