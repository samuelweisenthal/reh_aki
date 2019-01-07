import pandas


def split_dx_px(all_w_i):


	#Chris Albon to the rescue #http://chrisalbon.com/python/pandas_expand_cells_containing_lists.html

	cs = ['CPT4_PROCEDURES','DIAGNOSES','ICD9_PROCEDURES'] #drg not included since not list
	d_fs=[]
	for c in cs:
	    o_h_e = all_w_i[c].str.split().apply(pandas.Series)
	    o_h_e.rename(columns=lambda x: str(c)+'_'+str(x),inplace=True)
	    d_fs.append(o_h_e)
	f_Df= pandas.concat(d_fs,axis=1)

	f_Df.columns.values

	for c in cs:
	    del all_w_i[c]


	return_df =  pandas.concat([all_w_i,f_Df],axis=1)
	return return_df


