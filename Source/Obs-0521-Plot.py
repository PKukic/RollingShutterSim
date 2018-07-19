import numpy as np 
import matplotlib.pyplot as plt 
import re

import DataAnalysisTools as dat

# Directories where both the angular velocity data and the corrected angular velocity data are saved
camo_dir = '../Observations/0521/ang_vel_CAMO/'
rms_dir = '../Observations/0521/ang_vel_RMS/'
save = '../Graphs/Obs-0521/'
ang_dir = '../Observations/0521/angles/'

# Directory matrix
files_arr = [dat.findCAMO(camo_dir), dat.findNoCorr(rms_dir), dat.findTemporal(rms_dir), dat.findSpatial(rms_dir)]

print(files_arr[0])
print(files_arr[1])
print(files_arr[2])


def strip_rms(s):
	global rms_dir
	print(s)
	sub = re.findall(r'_.*?_.*?_(.*?)_.*?_.*?_.*?', s)[0]
	print(1, sub)
	return int(sub[:2])*3600 + int(sub[2:4])*60 + int(sub[4:6]) 

def strip_camo(s):
	global camo_dir
	print(s)
	sub = re.findall(r'_.*?_(.*?)_.*?', s)[0][:-1]
	print(3, sub)
	return int(sub[:2])*3600 + int(sub[2:4])*60 + int(sub[4:6])


# Sort the filename matrix by date of file
files_arr = (sorted(files_arr[0], key=strip_camo), sorted(files_arr[1], key=strip_rms), sorted(files_arr[2], key=strip_rms), sorted(files_arr[3], key=strip_rms))

n = len(files_arr[0])

for i in range(n):

	# Form filenames
	print(i)
	filename = save + str(i) + '.png'

	# Extract data
	time_camo = np.load(camo_dir + files_arr[0][i])['arr_0']
	time_nocorr = np.load(rms_dir + files_arr[1][i])['arr_0']
	time_temp = np.load(rms_dir + files_arr[2][i])['arr_0']
	time_spat = np.load(rms_dir + files_arr[3][i])['arr_0']

	av_camo = np.load(camo_dir + files_arr[0][i])['arr_1']
	av_nocorr = np.load(rms_dir + files_arr[1][i])['arr_1']
	av_temp = np.load(rms_dir + files_arr[2][i])['arr_1']
	av_spat = np.load(rms_dir + files_arr[3][i])['arr_1']

	phi = np.load(ang_dir + 'rms.npz')['arr_0'][i]

	name_rms = files_arr[0][i][:-4]
	name_camo = files_arr[3][i][:-4]
	print(name_rms, name_camo)

	print(phi)

	
	# Shift the time so that each time instance approximately matches the CAMO time
	referent = time_camo[0]

	deltat_nocorr = referent - time_nocorr[0]
	deltat_temp = referent - time_temp[0]
	deltat_spat = referent - time_spat[0]

	for i in range(len(time_nocorr)):
		time_nocorr[i] += deltat_nocorr
		time_temp[i] += deltat_temp
		time_spat[i] += deltat_spat


	# Plot the data
	plt.ioff()

	# plt.rc('text', usetex=True)
	# plt.rc('font', family='serif')
	
	plt.plot(time_nocorr, av_nocorr, 'ro', label = 'RMS uncorrected')
	plt.plot(time_temp, av_temp, 'go', label = 'RMS temporal')
	plt.plot(time_spat, av_spat, 'bo', markersize = 8, markerfacecolor = "None", markeredgecolor = "blue", markeredgewidth = 2, label = 'RMS spatial')
	plt.plot(time_camo, av_camo, 'y^', label = 'CAMO')
	
	plt.title(r'$\phi$ = {} [deg]'.format(phi))

	plt.legend(loc='best')

	plt.xlabel('Time [s]')
	plt.ylabel(r'$\omega$ ' + '[deg/s]')
	
	print(filename)

	plt.savefig(filename)

	plt.close()