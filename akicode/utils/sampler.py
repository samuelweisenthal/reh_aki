import numpy as np
import pandas
def independentSampler(df,dependencyVar):
    np.random.seed(0)#for replicating experiment
    '''Creates a dataframe that has independent samples from dataframe in which dependencyVar introduces a dependency
       For example, can call independentSampler(df,'PT_NUM') to get a random sample from df in which there are no dependencies
       due to the same patient having multiple admissions
    '''
    print "Starting with",len(df),"samples"
    print "Positive samples",len(df[df['target']==1])
    print "Negative samples",len(df[df['target']==0])
    print "Unique patients that generated positive samples",len(df[df['target']==1]['PATIENT_NUM'].unique())
    print "Unique patients that generated negative samples",len(df[df['target']==0]['PATIENT_NUM'].unique())
    print "Total unique patients",len(df['PATIENT_NUM'].unique())
    subSamples=[]
    unSampled=[]
    for el,g in df.groupby(dependencyVar): #for maybe a better soln, see http://stackoverflow.com/questions/22472213/python-random-selection-per-group
        c = np.random.choice(g.index); 
        subSamples.append(g.loc[[c]])
        was_sam=g.index.isin([c])
        unSampled.append(g[~was_sam])
    IndependentSamples=pandas.concat(subSamples)
    Unsampled=pandas.concat(unSampled)
    print "Now, have",len(IndependentSamples),"samples"
    print "Positive samples",len(IndependentSamples[IndependentSamples['target']==1])
    print "Negative samples",len(IndependentSamples[IndependentSamples['target']==0])
    print "Unique patients that generated positive samples",len(IndependentSamples[IndependentSamples['target']==1]['PATIENT_NUM'].unique())
    print "Unique patients that generated negative samples",len(IndependentSamples[IndependentSamples['target']==0]['PATIENT_NUM'].unique())
    print "Total unique patients",len(IndependentSamples['PATIENT_NUM'].unique())
    print
    print "UNsampled admissions",len(Unsampled)
    print "Positive UNsamples",len(Unsampled[Unsampled['target']==1])
    print "Negative UNsamples",len(Unsampled[Unsampled['target']==0])
    print "Unique patients that generated positive UNsamples",len(Unsampled[Unsampled['target']==1]['PATIENT_NUM'].unique())
    print "Unique patients that generated negative UNsamples",len(Unsampled[Unsampled['target']==0]['PATIENT_NUM'].unique())
    print "Total unique UNsampled patients",len(Unsampled['PATIENT_NUM'].unique())

    return (IndependentSamples,Unsampled)

def sampler(X,y,groups, random_state):
        #https://stackoverflow.com/questions/22472213/python-random-selection-per-group
        X['groupid'] = groups
        X['target'] = y 
        size = 1
        np.random.seed(random_state)
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace=False),:]
        X = X.groupby('groupid', as_index=False).apply(fn)
        print "len sampled", len(X['target'])
        return X[X.columns[:-2]], pandas.DataFrame(X['target']), pandas.DataFrame(X['groupid'])
