import numpy as np

def rem_rare(thresh,df): 
    
    '''Counts number of times variables observed and only 
    keeps those that are above some threshold.  Slightly 
    underestimates for continuous values which might be 
    zero and observed'''
    print len(df)
    print thresh
    if len(df) <= thresh: 
	print "LEN DF <= THRESH; so it will remove all columns..."
	exit()
    return df[df.columns[((((df > 0) & (df != np.NAN))*1).sum()>thresh).values]]

def rem_rare_wrap(thresh,df):
    '''wraps rem_rare and adds target back on. Inefficient.'''
    print "removing rare"
    d2 = rem_rare(thresh=thresh, df=df).copy()
    d2['target'] = df['target']
    return d2
