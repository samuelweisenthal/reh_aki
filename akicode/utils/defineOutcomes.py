'''Mostly get_expected_dx used for this study.
Gets outcomes for training (based on preprocessed data).'''

import numpy as np

any_in = lambda a,b:bool(set(a).intersection(set(b)))
#http://stackoverflow.com/questions/10668282/one-liner-to check-if-at-least-one-item-in-list-exists-in-another-list


def getDf(df):
    print "***Simply getting the dataframe***"
        
    df.replace(np.nan,'nan', regex=True,inplace=True)
    df['sampleNo']=range(0,len(df))
    df['target']=0

    return (df)

def get_unexpected_dx(df,Ptcol,Adcol,dxs,targets):

    '''If diagnosis is present in stay and NOT in past hospital stay, positive example. If was present in past hospital stay, EXCLUDE. (If was not present in past hospital stay, but was in some previous hospital stay, still count as positive.)  If AKI not present in current stay: if AKI present in previous stay, EXCLUDE. Else if AKI not present in current stay and not present in previous stay, negative example.  Target must be of the form '311.0' '''
    print get_unexpected_dx.__doc__
    print "targets",targets
    #make dataframes for positive and negative samples. Each needs sample number and will be grouped by that--not patient number  
    positiveSamples=[]        
    negativeSamples=[]
    sampleNumber=0
    df.sort_values([Ptcol,Adcol],inplace=True,axis=0)
    #df.replace(np.nan,'nan', regex=True,inplace=True) #don't do this, converts the whole df to string!
    print "not replacing"
    CumDxsDifferences = [] #diagnoses differences such that the most recent hospital stay is compared to diagnoses from all others
    MarDxDifferences = [] #diagnoses differences such that the most recent hosptial stay is compared to diagnoses from the previous hospital stay
    excludedAds=0
    excludedAds_neg =0
    dial_neg=0
    dial = 0
    for ptnum,admits in df.groupby(Ptcol):
        accumDxs=set()#this patient's accumulated diagnoses
        numReadmits=0
        for adid,admit in admits.groupby(Adcol):#sorted above, shouldn't need to be sorted 
            stayDxsSet = set(admit[dxs].iloc[[0]].values[0])
            CumDxsDiff = stayDxsSet - accumDxs #the set of new diagnoses from this visit (note that this is compared to accumulated dx, not last stay's dx)
            accumDxs = accumDxs | stayDxsSet #diagnoses now is union of set of past diagnoses with this stay's set of diagnose

            if numReadmits > 0: #we don't want this for the first stay because there aren't any features for it
                CumDxsDifferences.append(CumDxsDiff)
                #print "curent aki",admit['AKI_by_creat'].values[0]
                #print "curent aki t",admit['AKI_by_creat'].values[0]==1
                if any_in(targets,stayDxsSet) or admit['AKI_by_creat'].values[0]==1:#might be much faster to truncate codes and then just see if '584' is 'in' ...the dx_by_creat refers to diagnosis by labs
                    #set sample number of all previous readmits to the current sample number
                    previousStayDxsSet = set(admits.ix[admits.index[numReadmits-1]][dxs].values)
                    previousStayCreatDx = admits.ix[admits.index[numReadmits-1]]['AKI_by_creat']
                    #print "prev aki",previousStayCreatDx
                    #print "prev aki==1",previousStayCreatDx==1
                    if any_in(targets,previousStayDxsSet) or previousStayCreatDx==1:
                        excludedAds+=1
                    else:
                        if 'V45.11' in stayDxsSet :#only applies to AKI
                            dial+=1

                        posEx=admits.ix[admits.index[0]:admits.index[numReadmits-1]].copy()
                        posEx['sampleNo']=sampleNumber
                        posEx['target']=1
                        positiveSamples.append(posEx)
                        #break don't break because we want to count all future readmissions as + or - example
                        sampleNumber+=1
                else:
                    
                    previousStayDxsSet = set(admits.ix[admits.index[numReadmits-1]][dxs].values)
                    previousStayCreatDx = admits.ix[admits.index[numReadmits-1]]['AKI_by_creat']
                    if any_in(targets,previousStayDxsSet) or previousStayCreatDx==1:
                        excludedAds_neg+=1
                    else:
                        if 'V45.11' in stayDxsSet :#only applies to AKI
                            dial_neg+=1
                    
                        negEx=admits.ix[admits.index[0]:admits.index[numReadmits-1]].copy()
                        negEx['sampleNo']=sampleNumber
                        negEx['target']=0
                        negativeSamples.append(negEx)
                        
                        sampleNumber+=1
                    
                    
            numReadmits+=1
    print "Excluded",excludedAds,"AKI + admissions because there was a target, but there was a target in the DIRECTLY previous stay."
    print "Excluded",excludedAds_neg,"AKI - admissions because there wasn't a target, but there was a target in the DIRECTLY previous stay."
    print "Positive examples:",len(positiveSamples)
    print  "Dialysis, too (in +)", dial, "that's",dial/float(len(positiveSamples))*100,"percent"
    print  "Dialysis, too (in -)", dial_neg, "that's",dial_neg/float(len(positiveSamples))*100,"percent"
    print "Negative examples",len(negativeSamples)
    return (CumDxsDifferences,MarDxDifferences,positiveSamples,negativeSamples)


def get_unexpected_dx_memoryless(df,Ptcol,Adcol,dxs,targets):

    '''Note first that the name is slightly off-- actually we exclude if past whether the current stay is negative or positive. If diagnosis is present in stay and NOT in past hospital stay, positive example. If was present in past hospital stay, EXCLUDE. (If was not present in past hospital stay, but was in some previous hospital stay, still count as positive.)  If AKI not present in current stay: if AKI present in previous stay, EXCLUDE. Else if AKI not present in current stay and not present in previous stay, negative example.  Target must be of the form '311.0' '''
    print get_unexpected_dx_memoryless.__doc__
    print "targets",targets
    #make dataframes for positive and negative samples. Each needs sample number and will be grouped by that--not patient number  
    positiveSamples=[]        
    negativeSamples=[]
    sampleNumber=0
    df.sort_values([Ptcol,Adcol],inplace=True,axis=0)
    #df.replace(np.nan,'nan', regex=True,inplace=True)
    CumDxsDifferences = [] #diagnoses differences such that the most recent hospital stay is compared to diagnoses from all others
    MarDxDifferences = [] #diagnoses differences such that the most recent hosptial stay is compared to diagnoses from the previous hospital stay
    excludedAds=0
    excludedAds_neg =0
    dial_neg=0
    dial = 0
    for ptnum,admits in df.groupby(Ptcol):
        accumDxs=set()#this patient's accumulated diagnoses
        numReadmits=0#a list of this patient's diagnoses from each stay
        for adid,admit in admits.groupby(Adcol):#sorted above, shouldn't need to be sorted 
            stayDxsSet = set(admit[dxs].iloc[[0]].values[0])
            CumDxsDiff = stayDxsSet - accumDxs #the set of new diagnoses from this visit (note that this is compared to accumulated dx, not last stay's dx)
            accumDxs = accumDxs | stayDxsSet #diagnoses now is union of set of past diagnoses with this stay's set of diagnose

            if numReadmits > 0: #we don't want this for the first stay because there aren't any features for it
                CumDxsDifferences.append(CumDxsDiff)
                #print "curent aki",admit['AKI_by_creat'].values[0]
                #print "curent aki t",admit['AKI_by_creat'].values[0]==1
                if any_in(targets,stayDxsSet) or admit['AKI_by_creat'].values[0]==1:#might be much faster to truncate codes and then just see if '584' is 'in' ...the dx_by_creat refers to diagnosis by labs
                    #set sample number of all previous readmits to the current sample number
                    previousStayDxsSet = set(admits.ix[admits.index[numReadmits-1]][dxs].values)
                    previousStayCreatDx = admits.ix[admits.index[numReadmits-1]]['AKI_by_creat']
                    #print "prev aki",previousStayCreatDx
                    #print "prev aki==1",previousStayCreatDx==1
                    if any_in(targets,previousStayDxsSet) or previousStayCreatDx==1:
                        excludedAds+=1
                    else:
                        if 'V45.11' in stayDxsSet :#only applies to AKI
                            dial+=1

                        posEx=admits.ix[[admits.index[numReadmits-1]]].copy()
                        posEx['sampleNo']=sampleNumber
                        posEx['target']=1
                        positiveSamples.append(posEx)
                        #break don't break because we want to count all future readmissions as + or - example
                        sampleNumber+=1
                else:
                    
                    previousStayDxsSet = set(admits.ix[admits.index[numReadmits-1]][dxs].values)
                    previousStayCreatDx = admits.ix[admits.index[numReadmits-1]]['AKI_by_creat']
                    if any_in(targets,previousStayDxsSet) or previousStayCreatDx==1:
                        excludedAds_neg+=1
                    else:
                        if 'V45.11' in stayDxsSet :#only applies to AKI
                            dial_neg+=1
                    
                        negEx=admits.ix[[admits.index[numReadmits-1]]].copy()
                        negEx['sampleNo']=sampleNumber
                        negEx['target']=0
                        negativeSamples.append(negEx)
                        
                        sampleNumber+=1
                    
                    
            numReadmits+=1
    print "Excluded",excludedAds,"AKI + admissions because there was a target, but there was a target in the DIRECTLY previous stay."
    print "Excluded",excludedAds_neg,"AKI - admissions because there wasn't a target, but there was a target in the DIRECTLY previous stay."
    print "Positive examples:",len(positiveSamples)
    print  "Dialysis, too (in +)", dial, "that's",dial/float(len(positiveSamples))*100,"percent"
    print  "Dialysis, too(in -)", dial_neg, "that's",dial_neg/float(len(positiveSamples))*100,"percent"
    print "Negative examples",len(negativeSamples)
    return (CumDxsDifferences,MarDxDifferences,positiveSamples,negativeSamples)

def get_expected_dx(df,Ptcol,Adcol,dxs,targets,memoryless=0):
    
    '''If AKI in current stay, positive sample. If AKI not in current stay, negative sample. Does NOT exclude cases where AKI in most recent stay. Target must be of the form '311.0' '''
    print get_expected_dx.__doc__
    print "targets",targets
    print
    print "**Memoryless**:",memoryless
    print
    #make dataframes for positive and negative samples. Each needs sample number and will be grouped by that--not patient number  
    positiveSamples=[]        
    negativeSamples=[]
    sampleNumber=0
    df.sort_values([Ptcol,Adcol],inplace=True,axis=0)
    #df.replace(np.nan,'nan', regex=True,inplace=True)
    dial_neg=0
    dial = 0
    for ptnum,admits in df.groupby(Ptcol):
        numReadmits=0#a list of this patient's diagnoses from each stay
        for adid,admit in admits.groupby(Adcol):#sorted above, shouldn't need to be sorted 
            stayDxsSet = set(admit[dxs].iloc[[0]].values[0])
            if numReadmits > 0: #we don't want this for the first stay because there aren't any features for it
                if any_in(targets,stayDxsSet) or admit['AKI_by_creat'].values[0]==1:#might be much faster to truncate codes and then just see if '584' is 'in' ...the dx_by_creat refers to diagnosis by labs
                    #set sample number of all previous readmits to the current sample number
                    if 'V45.11' in stayDxsSet :#only applies to AKI
                        dial+=1
                    if not memoryless:
                        posEx=admits.ix[admits.index[0]:admits.index[numReadmits-1]].copy() #get all previous admissions
                    if memoryless:
                        posEx=admits.ix[[admits.index[numReadmits-1]]].copy() #get most recent admission
                    posEx['sampleNo']=sampleNumber
                    posEx['target']=1
                    positiveSamples.append(posEx)
                    sampleNumber+=1
                else:
                    
                    if 'V45.11' in stayDxsSet :#only applies to AKI
                        dial_neg+=1
                    if not memoryless:
                        negEx=admits.ix[admits.index[0]:admits.index[numReadmits-1]].copy() #get all previous admissions
                    if memoryless:
                        negEx=admits.ix[[admits.index[numReadmits-1]]].copy() #get most recent admission
                        
                    negEx['sampleNo']=sampleNumber
                    negEx['target']=0
                    negativeSamples.append(negEx)
                    sampleNumber+=1
                    
                    
            numReadmits+=1
    print "Positive examples:",len(positiveSamples)
    print "Dialysis, too (in +)", dial, "that's",dial/float(len(positiveSamples))*100,"percent"
    print "Dialysis, too (in -)", dial_neg, "that's",dial_neg/float(len(positiveSamples))*100,"percent"
    print "Negative examples",len(negativeSamples)
    
    return (positiveSamples,negativeSamples)
    
