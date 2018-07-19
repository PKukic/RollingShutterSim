import DataAnalysisTools as dat 

data_dir = '../Observations/0521/camo_20180521/'
data_name_arr = dat.findTXT(data_dir)
save_dir = '../Observations/0521/ang_vel_CAMO/'

for name in data_name_arr:
	dat.CAMOtoAVT(data_dir, name, save_dir)

