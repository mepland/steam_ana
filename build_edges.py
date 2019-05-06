#!/usr/bin/env python
import pandas as pd
import itertools

fpath = '../data/games_1.csv'
out_path = './'
chunksize = 10**5

for i,df in enumerate(pd.read_csv(fpath,usecols=[0,1],dtype={'steamid':object,'appid':int},chunksize=chunksize)):
    df = df.drop_duplicates()

    # Create the edges between games / nodes (appid) from each player (steamid)
    all_edge_rows_list = []
    for steamid in df['steamid'].unique():
        df_steamid = df[(df['steamid']==steamid)]

        # save out all the edge combinations we can make from the appid with itertools.combinations
        for edge in list(itertools.combinations(df_steamid['appid'], 2)):
            # always make n1 the low number, n2 the high number, will reduce duplicates, save space
            n1 = min(edge)
            n2 = max(edge)
            all_edge_rows_list.append({'n1':n1,'n2':n2})

    # create df_edges
    df_edges = pd.DataFrame(all_edge_rows_list)
    # merge up duplicates
    df_edges = df_edges.groupby(df_edges.columns.tolist()).size().reset_index().rename(columns={0:'records'})
    # save out
    df_edges.to_csv('{0:s}edges_{1:d}.csv'.format(out_path,i), index=False, na_rep='nan')
