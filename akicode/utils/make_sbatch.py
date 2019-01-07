'''This is a work in progress, not implemented.  Wanted to
automate the sbatch creation to be able to run all experiments
automatically in sequence.  As is, have to call individual sbatch
files to run each.  One issue is that there is a 5 day limit on
jobs running, so would be difficult.  Will potentially pick this up in the future.'''

def make_sbatch(sbatch_file, sbatch_errors, partition, run_time, nodes, memory, iterations, output_dir,
		fof_file, fof_file_errors, skgf, ons, nj, cal3, 
		x_path, y_path, g_path, dx_path,
		weight, samp):
	
	def setup(sbatch_file, partition, run_time, nodes, memory, iterations, output_dir):

		f = open(sbatch_file, 'w')
		
		f.write('#!/bin/sh \n')
		f.write('#SBATCH --partition=' + partition + '\n')
		f.write('#SBATCH --time=' + run_time + '\n')
		f.write('#SBATCH -N ' + nodes + '\n')
	 	f.write('#SBATCH --mem=' + memory + '\n')
		f.write('#SBATCH --mail-type=begin' + '\n')
		f.write('#SBATCH --mail-type=end' + '\n')
		f.write('#SBATCH -a 1-' + iterations + '\n')
		f.write('#SBATCH -o' + output_dir + '/log%a.txt' + '\n')
		f.write('module load anaconda' + '\n')
		return f

	def make_fof(f, output_dir, fof_file,  skgf, ons, 
		nj, cal3, x_path, y_path, g_path, weight, samp, 
	        alpha=None):
		f.write('python ' + fof_file
			+ ' -rs $SLURM_ARRAY_TASK_ID'
			+ ' -skgf ' + skgf 
			+ ' -ons ' + ons 
			+ ' -nj ' + nj
			+ ' -cal3 ' + cal3 
			+ ' -x ' + x_path 
			+ ' -y ' + y_path
			+ ' -g ' + g_path 
			+ ' -o ' + output_dir 
			+ ' -weight ' + weight 
			+ ' -samp ' + samp)
		if alpha:
			f.write('-alpha ' + str(alpha))
		f.write('\n')

	f = setup(sbatch_file=sbatch_file, partition=partition, run_time=run_time,
		nodes=nodes, memory=memory, iterations=iterations, output_dir=output_dir) 
	make_fof(f=f, output_dir=output_dir, fof_file=fof_file,
		skgf=skgf, ons=ons, nj=nj, cal3=cal3,
		x_path=x_path, y_path=y_path, g_path=g_path, 
		weight=weight, samp=samp)

	f.write('python analyze_cv_data_clean.py ' + output_dir+'/ ' 
		+ '-x ' + x_path + ' ' 
		+ '-y ' + y_path + ' '
		+ '-g ' + g_path + ' '
		+ '-d ' + dx_path + '\n')

	f.close() 

        f = setup(sbatch_file=sbatch_file, partition=partition, run_time=run_time,
                nodes=nodes, memory=memory, iterations=iterations, output_dir=output_dir)
	#these are from analyze_cv_results_clean. 
	#should really save to file or pass not remake here
	dxs = ['DX_','ETHN_','GENDER_','AGE_']*2	
	alphas = [1e-5]*4+[0.015]+[0]*3
	for outcome, tag_oi, alpha in zip([0,0,0,0,1,1,1,1], dxs, alphas):
		if outcome == 1: dirname = 'case_error'	
		else: dirname = 'contr_error'
		my_dir = output_dir + dirname + '/' + tag_oi + '/'    
		y_path = my_dir + 'y_error.csv'
		X_path = my_dir + 'X_error.csv'
		g_path = my_dir + 'group_error.csv'
		make_fof(f=f, output_dir=my_dir, fof_file=fof_file_errors,
			skgf=skgf, ons=ons, nj=nj, cal3=cal3,
			x_path=X_path, y_path=y_path, g_path=g_path,
			weight=weight, samp=samp, alpha=alpha)

	f.close()


#make_sbatch(sbatch_file='testsb.sbatch',
#	partition='preempt', run_time='1-00:00:00',
#	nodes='1',
#	memory='100gb',
#	iterations='50',
#	output_dir='/homedirec/user/fof_full3QFit_dialE_ex_ED',
#	fof_file='fof2.py',skgf='0', ons='1', nj='1', cal3='0', 
#	x_path='/homedirec/user/X_aki_full_dialE_ex_ED.csv',
#	y_path='/homedirec/user/y_aki_full_dialE_ex_ED.csv',
#	g_path='/homedirec/user/groups_aki_full_dialE_ex_ED.csv', 
#	weight='0', samp= '0')
