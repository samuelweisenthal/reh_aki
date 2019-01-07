'''For summary statistics'''

def get_nhosp_per_id(agg_res):
    print "#hosp", len(agg_res)
    ids = agg_res[1]
    unique_ids = ids.unique()
    print "#pt", len(unique_ids)
    pt_to_nhosp = {el:len(agg_res[agg_res[1]==el]) 
               for el in unique_ids}
    return pt_to_nhosp
