
''''#Get contingency matrix with ESKD excluded
#Again, check total # from sCr'''

import pandas
dxs = pandas.read_csv('/homedirec/user/or_and_byCreat_byAdm.csv', index_col=0)
dial = pandas.read_csv('/homedirec/user/DialEx.csv', index_col=0)
from_labs = pandas.read_csv('/homedirec/user/dx_from_labs_fixed.csv', index_col=0)

print len(from_labs)
print from_labs.sum()

j = dxs.join(dial, how='inner') #inner join to remove peds
j = j.join(from_labs, how='left')

j = j[j['keep']==1] #remove dial

j['AKI_by_creat'].fillna(0).sum()

j['keep'].sum()

print pandas.crosstab(j.AKI_by_admi>0, j.AKI_by_creat>0, margins=True)

print j['or'].sum()

print j['or'].sum()/float(len(j))*100

