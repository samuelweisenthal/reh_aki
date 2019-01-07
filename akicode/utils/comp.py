'''Requires python3.  Depends on Bayescomp library. For comparing systems'''

import pickle
import baycomp 
import pdb
import numpy as np
import matplotlib
#import scipy.stats.mstats.normaltest as nml
import scipy
import matplotlib as mpl
import argparse
parser = argparse.ArgumentParser(description='run w plot?')
parser.add_argument('-ps', '--plot_stuff', type=int)
args = parser.parse_args()
plot_stuff = args.plot_stuff
print("plotting?", plot_stuff)
if plot_stuff:
	mpl.use('Agg')
	import matplotlib.pyplot as plt


homedirec_loc = '/homedirec/user/'
_dir1 = '/homedirec/user/fof_full3QFit_dialE_lassoPen/'
_dir2 = '/homedirec/user/fof_full3QFit_dialE_lasso/'

def get_met_from_dir(_dir):
	met_d = {}
	mets = ['roc', 'pr', 'bri']
	for met in mets:
		met_d[met] = []
		
	for i in range(1,51):
		fname = _dir + 'data_rs_'+str(i) + '.pickle'
		with open(fname, 'rb') as f: d = pickle.load(f, encoding='latin1')
		[best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, 
		best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co, 
		feat_names, _] = d
		for met in mets:
			met_d[met].extend(m_d_outers[met])
	nmet_d = {}
	for met, el in met_d.items():
		nmet_d[met] = np.array(met_d[met])
		#p = nml(nmet_d[met])
		#print "p nml?", p
		if plot_stuff:
			plt.hist(nmet_d[met])
			xl = met + _dir
			plt.xlabel(xl)
			sn = _dir + str(met) + 'hist.png'
			plt.savefig(sn)
			sn2 = '/homedirec/user/dists/'+ _dir[18:-1] + '_' + met + '.png'
			plt.savefig(sn2)  
			plt.clf()
			myplot = plt
			scipy.stats.probplot(nmet_d[met], plot=myplot)
			sn2 = '/homedirec/user/dists/'+ _dir[18:-1] + '_' + met + 'qq.png'
			plt.savefig(sn2)
			plt.clf() 
	return nmet_d


clf_dir = [('GBC','fof_full3QFit_dialE/'),
('LR1','fof_full3QFit_dialE_lasso/'),
('HPLR1','fof_full3QFit_dialE_lassoPen/'),
('WGBC','fof_full3QFit_dialE_weight/'),
('WLR1','fof_full3QFit_dialE_lasso_weight/'),
('WHPLR1','fof_full3QFit_dialE_lassoPen_weight/'),
('SGBC','fof_full3QFit_dialE_samp/'),
('SLR1','fof_full3QFit_dialE_lasso_samp/'),
('SHPLR1','fof_full3QFit_dialE_lassoPen_samp/'),
('RGBG', 'fof_full3QFit_dialE_memoryless/'),
('MGBC','fof_full3QFit_dialE_meds/'),
('MLR1','fof_full3QFit_dialE_meds2/'),
('NGBC','fof_full3QFit_dialE_shuffle_y/')]

def compare_dir(_dir1, _dir2):
	m1 = get_met_from_dir(_dir1)
	m2 = get_met_from_dir(_dir2)
	ropes = [0.01, 0.001, 0.01]
	mets = ['roc', 'bri', 'pr']
	res = {}
	res_np = {}
	for met,rope in zip(mets, ropes):
		#print("met", met)
		res[met] = baycomp.two_on_single(m1[met], m2[met], rope=rope, runs=50)
	return res

res = compare_dir(_dir1, _dir2)

import pandas
clfs = [el[0] for el in clf_dir]
mets = ['roc', 'pr', 'bri']
dfs = {}

for met in mets:
	dfs[met] = pandas.DataFrame(columns=clfs, index=clfs)	
	for i, (clf1, _dir1) in enumerate(clf_dir):
		for j, (clf2, _dir2) in enumerate(clf_dir[i+1:]):
			dir1 = homedirec_loc + _dir1
			dir2 = homedirec_loc + _dir2
			mc = compare_dir(dir1, dir2)
		 	
			#from stackoverflow https://stackoverflow.com/questions/40071006/python-2-7-print-a-dictionary-without-brackets-and-quotation-marks
			#tab.loc[clf1, clf2] = (';'.join("{}: {}".format(k, [round(el,2) for el in v]) for k, v in mc.items())).replace('[','').replace(']',' ')
			dfs[met].loc[clf1, clf2] = str([round(el,2) for el in mc[met]])[1:-1]
	pdb.set_trace()
	dfs[met] = dfs[met].drop(columns=['GBC'])
with open('comp.txt', 'w') as f:
	for met in mets:
		f.write(met)
		f.write(dfs[met].to_latex())
		f.write('\n')
import pickle
with open('comp', 'w') as f:
	pickle.dump(dfs, f)


pdb.set_trace()
