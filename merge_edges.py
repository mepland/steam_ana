#!/usr/bin/env python
import pandas as pd
import glob

fpath = './edges_*.csv'
# fpath = '../data/edges/edges_*.csv'
out_path = '../'

df = None
for f in sorted(glob.glob(fpath)):
    print(f)
    df_new = pd.read_csv(f,dtype={'n1':int,'n2':int,'records':int})

    if df is None:
        df = df_new.copy()
    else:
        df = df.merge(df_new,on=['n1', 'n2', 'records'], how='outer').groupby(['n1', 'n2'], as_index=False)['records'].sum()

# clean
df = df.drop_duplicates()

# save out
df.to_csv('{0:s}edges_merged.csv'.format(out_path), index=False, na_rep='nan')
