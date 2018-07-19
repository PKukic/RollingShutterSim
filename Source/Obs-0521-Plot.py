import numpy as np 
import matplotlib.pyplot as plt 
import os
import re

# Directories where both the angular velocity data and the corrected angular velocity data are saved
RMS_NOCORR_DIR =  '../Data/0521/CA0001_20180521_012018_461559/ang_vel_nocorr/'
RMS_TEMP_DIR = '../Data/0521/CA0001_20180521_012018_461559/ang_vel_temp/'
RMS_SPAT_DIR = '../Data/0521/CA0001_20180521_012018_461559/ang_vel_spat/'
CAMO_DIR = '../Data/0521/camo_20180521/ang_vel/'

# Directories where the plots are saved
SAVE = '../Plots/0521/all/'
ANG_DIR = '../Data/0521/angles/'

# Directory matrix
DIR_ARR = [RMS_NOCORR_DIR, RMS_TEMP_DIR, RMS_SPAT_DIR, CAMO_DIR]

# Files matrix
files_arr = [[], [], [], []]

# Get all files
for i in range(4):
	for root, dirs, files in os.walk(DIR_ARR[i]):
		for file in files:
			if file.endswith('.npz'):
				files_arr[i].append(DIR_ARR[i] + file)

# print(files_arr)

n = len(files_arr[0])
print(len(files_arr[0]), len(files_arr[1]), len(files_arr[2]), len(files_arr[3]))

# for i in range(n):
	# print(files_arr[0][i], files_arr[1][i], files_arr[2][i], sep = '\t')

def strip_nocorr(s):
	global RMS_NOCORR_DIR
	s = s[len(RMS_NOCORR_DIR)::]
	sub = re.findall(r'_.*?_.*?_(.*?)_.*?_.*?_.*?', s)[0]
	print(1, sub)
	return int(sub[:2])*3600 + int(sub[2:4])*60 + int(sub[4:6]) 

def strip_corr(s):
	global RMS_TEMP_DIR
	s = s[len(RMS_TEMP_DIR)::]
	sub = re.findall(r'_.*?_.*?_(.*?)_.*?_.*?_.*?', s)[0]
	print(2, sub)
	return int(sub[:2])*3600 + int(sub[2:4])*60 + int(sub[4:6])

def strip_camo(s):
	global CAMO_DIR
	s = s[len(CAMO_DIR)::]
	sub = re.findall(r'_(.*?)_.*?', s)[0][:-1]
	print(3, sub)
	return int(sub[:2])*3600 + int(sub[2:4])*60 + int(sub[4:6])


# Sort the filename matrix by date of file
files_arr = (sorted(files_arr[0], key=strip_nocorr), sorted(files_arr[1], key=strip_corr), sorted(files_arr[2], key=strip_corr), sorted(files_arr[3], key=strip_camo))


for i in range(n):

	# Form filenames
	print(i)
	filename = SAVE + str(i) + '.png'

	# Extract data
	time_nocorr = np.load(files_arr[0][i])['arr_0']
	time_temp = np.load(files_arr[1][i])['arr_0']
	time_spat = np.load(files_arr[2][i])['arr_0']
	time_camo = np.load(files_arr[3][i])['arr_0']

	av_nocorr = np.load(files_arr[0][i])['arr_1']
	av_temp = np.load(files_arr[1][i])['arr_1']
	av_spat = np.load(files_arr[2][i])['arr_1']
	av_camo = np.load(files_arr[3][i])['arr_1']

	phi = np.load(ANG_DIR + 'rms.npz')['arr_0'][i]

	name_rms = files_arr[0][i][len(RMS_NOCORR_DIR):-4]
	name_camo = files_arr[3][i][len(CAMO_DIR):-4]

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