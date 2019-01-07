import pdb
from subprocess import call
f = open('analysis_files.txt').read() 
for my_file in f.split('\n')[:-1]:
	try:
		call(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors", "--ExecutePreprocessor.timeout=-1", "--inplace", my_file])
	except:
		print "didn't work", my_file

