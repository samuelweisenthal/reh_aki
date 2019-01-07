# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:57:01 2016
Some functions to remove special characters.
"""
import re
# def StrToStrIntorStr(x):
#     try:
#         return str(int(float(x)))
#     except:
#         return x
        
def removeSpecCharFromDF(df):   
#removes sc from all of the values in the dataset, not the columns--for col, see below     
        for el in df.columns.values:
            df[el]=df[el].astype(str)
            df[el] = df[el].str.replace('[^\w\s]','')
            
            
def remSpecCharFromColumns(d):
    #Note that, despite the name, it doesn't remove [],(),_
    d=d.rename(columns = {c:re.sub('[^a-zA-Z0-9\n\_\[\]\)\( ]','',c) for c in d.columns})
    return d   

def remDot(s):
    return re.sub('\.','',s)
    
def ds(patients,DigitsToTake):  
    #old function, don't use          
    coi = patients[patients.columns.values[75:129]]

    for el in coi.columns.values:
        coi[el] = coi[el].str.replace('[^\w\s]','')
    
    
    for el in coi.columns.values:
        coi[el] = coi[el].str[0:DigitsToTake]
    
   
    return coi
    
    
#coi = pandas.get_dummies(coi,prefix_sep='=')
#coi = coi.groupby(lambda x:x,axis=1).sum()
