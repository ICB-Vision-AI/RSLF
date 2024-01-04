from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Useful_functions as uf
import csv
import os

def im2data(name, mvt_num, aff=False, file_adress = "LFs", size_mi = 9, size_sa = 512):
	'''
	Extract images points from Light Fields and save them in a csv file.

	Notes
	-----
	You can change the parameters in cv2.goodFeaturesToTrack() and flow_params to adapt to your needs.

	Parameters
	----------
	name: str
		The name of the file where the light field is.
	mvt_num: str
		The velocity scenario number associated with the light field.
	aff: bool
		If True, plot some figures.
	file_adress: str
		The adress of the file where the LF files are, default is "LFs"
	size_mi: int
		size of the square array of images, default is 9.
	size_sa: int
		size of the images, default is 512.

	Returns
	-------
	Return 0

	'''

	### --- Load the light field from the images ---

	data_file = file_adress+'/'+name+'/'

	# Prepare an array to contain the 4D light field
	data = np.zeros((size_mi, size_mi, size_sa, size_sa, 3))
	
	# Go through all the images constituting the light field, convert them to rgb and fill the array "data".
	for i in range(size_mi):
		for j in range(size_mi):
			img = cv2.imread(data_file + "Mvt_"+str(mvt_num)+"/Cam_" +str(i) + '_'+str(j)+'.png')
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
			data[i,j,:,:,:] = img
	
	#Rearange the array "data", so it is organized as, (u, v, s, t), t being the line of micro-image for example
	data = np.swapaxes(data, 0,1)
	data = np.swapaxes(data, 2,3)
	
	print('\n LF', name+"_Mvt_"+str(mvt_num), "of size:", data.shape)
	
	if 0:
		#plot central SAI if wanted
		plt.imshow(data[int(data.shape[0]/2),int(data.shape[1]/2)])
		plt.show()
	
	
	### --- Feature points detection ---
	
	print('\n Corner detection 1/2')	
	# Extracts features in the central SAI and uses optical flow to estimate their positions in the other SAIs
	
	corners = []
	central_sa = cv2.cvtColor((data[int(data.shape[0]/2),int(data.shape[0]/2),:,:]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
	central_pts = cv2.goodFeaturesToTrack(central_sa, maxCorners=5000, qualityLevel=0.01, minDistance=0)
	num=0
	for pt in central_pts:
		x, y = pt.ravel()
		corners.append(np.array([i, j, y, x, num]))
		num+=1
	
	# Define the parameters for dense optical flow estimation
	flow_params = dict(pyr_scale=0.5, levels=3, winsize=50, iterations=9, poly_n=7, poly_sigma=1.1, flags=0)
	
	# Go through every SAIs to compute the optial flow between them and the central SAI
	for j in range(data.shape[0]):
		for i in range(data.shape[1]):
			print(' Progress:', int((j*data.shape[1]+i)/(data.shape[0]*data.shape[1])*100), end="%\r")
			
			sa_ij = cv2.cvtColor((data[i,j,:,:]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
			
			# Estimate the optical flow between the two images
			flow1 = cv2.calcOpticalFlowFarneback(central_sa, sa_ij, None, **flow_params)
			fx = cv2.remap(flow1[:,:,0], central_pts[:,0,0], central_pts[:,0,1], cv2.INTER_LINEAR, borderValue=np.nan)
			fy = cv2.remap(flow1[:,:,1], central_pts[:,0,0], central_pts[:,0,1], cv2.INTER_LINEAR, borderValue=np.nan)
			
			coords = []
			num=0
			for num_pt in range(central_pts.shape[0]):
				# Compute the position of the feature points in the other SAIs
				x2, y2 = central_pts[num_pt,0,0] + fx[num_pt,0], central_pts[num_pt,0,1] + fy[num_pt,0]
				if x2>=0 and x2<size_sa-1 and y2>=0 and y2<size_sa-1:
					# If they are still inside the SAI
					coords.append(np.array([y2, x2]))
					corners.append(np.array([i, j, y2, x2, num]))
				num+=1
			coords = np.array(coords)
			
			if 0:
				# Plot the SAIs if wanted
				fig, ax = plt.subplots()
				ax.imshow(cv2.transpose(data[i,j,:,:]))
				ax.plot(coords[:, 0], coords[:, 1], color='red', marker='o', linestyle='None', markersize=6)
				plt.show()
	
	corners = np.array(corners)
	print(' Progress: Finished       ')
	
	
	### --- Estimate the position of the feature points in the micro-images ---
	
	print('\n Corner detection 2/2')
	# Interpolate the points in the Pseudo-Epipolar Plane and sample at interger s and t
	
	corners_ = []
	y_pred = []
	corners_temp1 = []
	new_y=-1
	
	# For every 3D point, we want to resample the image points at interger s
	for y in range(int(max(corners[:,4])+1)):
		pnt = corners[np.where(corners[:,4]==y)]
		
		s_min =  round(np.min(pnt[:,2]))
		s_max =  round(np.max(pnt[:,2]))
		t_min =  round(np.min(pnt[:,3]))
		t_max =  round(np.max(pnt[:,3]))
		
		new=0
		
		# Go through every (non-int) t
		for t in np.arange(t_min, t_max):
			pnt_s_y = pnt[np.where(abs(pnt[:,3]-t)<1),0][0]
			pnt_s_x = pnt[np.where(abs(pnt[:,3]-t)<1),2][0]
			
			if pnt_s_x.shape[0]>5:
				#If enough points
				new+=1
				if new==1:
					new_y+=1	
				# Fit a curve in the Pseudo-Epipolar Plane
				z = np.poly1d(np.polyfit(pnt_s_x, pnt_s_y, 3))
				
				# Detect outliers by detecting the points that are to far from the fitted curve
				diff = abs(pnt_s_y-z(pnt_s_x))
				inliers =  np.where(diff<2)[0]
				
				# Redo the fitting with only inliers
				if inliers.shape[0]<pnt_s_x.shape[0] and inliers.shape[0]>5:
					z = np.poly1d(np.polyfit(pnt_s_x[inliers], pnt_s_y[inliers], 3))
				
				if 0:
					# Plot the feature points in the Pseudo-Epilpolar Plane in wanted.
					xp = np.linspace(s_min, s_max, 100)
					plt.plot(pnt_s_x, pnt_s_y, '.', xp, z(xp), '-')
					plt.show()
				
				for pt_ in pnt[np.where(abs(pnt[:,3]-t)<1)]:
					corners_temp1.append(np.array([z(round(pt_[2])), pt_[1], round(pt_[2]), pt_[3], new_y]))
	
	corners_temp1 = np.array(corners_temp1)
	corners_temp2 = []
	new_y=-1
	
	# Redo the same for resampling at interger t
	for y in range(int(max(corners_temp1[:,4])+1)):
		pnt = corners_temp1[np.where(corners_temp1[:,4]==y)]
		s_min =  round(np.min(pnt[:,2]))
		s_max =  round(np.max(pnt[:,2]))
		t_min =  round(np.min(pnt[:,3]))
		t_max =  round(np.max(pnt[:,3]))
		new=0
		for s in np.arange(s_min, s_max):
			pnt_t_y = pnt[np.where(pnt[:,2]==s),1][0]
			pnt_t_x = pnt[np.where(pnt[:,2]==s),3][0]
			
			if np.unique(pnt_t_y).shape[0]>4:
				new+=1
				if new==1:
					new_y+=1
				z = np.poly1d(np.polyfit(pnt_t_x, pnt_t_y, 3))
				diff = abs(pnt_t_y-z(pnt_t_x))
				inliers =  np.where(diff<2)[0]
				if inliers.shape[0]<pnt_t_x.shape[0] and inliers.shape[0]>5:
					z = np.poly1d(np.polyfit(pnt_t_x[inliers], pnt_t_y[inliers], 3))
				if 0:
					xp = np.linspace(t_min, t_max, 100)
					plt.plot(pnt_t_x, pnt_t_y, '.', xp, z(xp), '-')
					plt.show()
				for pt_ in pnt[np.where(pnt[:,2]==s)]:
					corners_temp2.append(np.array([pt_[0], z(round(pt_[3])), pt_[2], round(pt_[3]), new_y]))
	corners = np.array(corners_temp2)
	
	
	### --- Mean the redundant points ---
	
	# For every 3D points,
	for y in range(int(max(corners[:,4])+1)):
		pnt = corners[np.where(corners[:,4]==y)]
		
		# For every micro-image it is projeted in,
		for s in np.unique(pnt[:,2]):
			for t in np.unique(pnt[:,3]):
				c=0
				mean_cor = []
				for cor in pnt:
					if cor[2] == s and cor[3]==t:
						mean_cor.append(np.array([cor[0], cor[1]]))
						c=1
				if c==1:
					# If it is seen at least one time, mean the positions of the projected points
					mean_cor = np.mean(np.array(mean_cor), axis=0)
					if 0:
						# Plot the micro-image and the mean position if wanted
						plt.imshow(cv2.transpose(data[:,:,int(s),int(t)]))
						plt.plot(mean_cor[0], mean_cor[1], color='red', marker='+',
											linestyle='None', markersize=6)
						plt.show()
					corners_.append(np.array([mean_cor[0], mean_cor[1], s, t]))
					y_pred.append(y)
	
	corners = np.array(corners_)
	y_pred = np.array(y_pred)

	print(' Progress: Finished       ')
	
	
	### --- Save the computed data ---
	
	nb_points = int(np.max(y_pred+1))
	print('\n Nb points: ', nb_points)

	if aff:
		plt.figure()
		plt.imshow(cv2.transpose(data[int(data.shape[0]/2),int(data.shape[0]/2)]))
		plt.scatter(corners[:,2],corners[:,3],c = y_pred, cmap='nipy_spectral')
		plt.show()

	# Load intrisic parameters
	calib_param = uf.load_calib_param()
	F, D, O, f, s, t, pix = calib_param

	m_rslf = []
	
	# Convert from pixel to metric with intrinsic parameters
	for k in range(nb_points):
		from_k = np.where(y_pred == k)[0]
		for j in range(corners[from_k].shape[0]):
			m = np.array([corners[from_k][j,0]*pix-s[1]/2, corners[from_k][j,1]*pix-t[1]/2, corners[from_k][j,2], corners[from_k][j,3], k])
			m_rslf.append(m)

	if not os.path.exists('Data/Data_'+ name + '_Mvt_'+str(mvt_num)):
		# Create a new directory because it does not exist
		os.makedirs('Data/Data_'+ name+ '_Mvt_'+str(mvt_num))
		print("New directory created!")

	# Write the coordinates of the projected points in a csv file
	with open('Data/Data_'+ name + '_Mvt_'+str(mvt_num) +'/rslf_image.csv', 'w') as f_:
		writer = csv.writer(f_)
		for i in range(len(m_rslf)):
			writer.writerows(np.array([m_rslf[i]]))
	return 0

if __name__ == "__main__":
	if 0:
		# One file specifically
		name = "rabbit"
		mvt_num = 10
		im2data(name, mvt_num, aff=True)
	else:
		# List of files and mvt scenario
		
		#names = ["chart", "rabbit", "table", "bedroom", "couch", "fireplace", "living_room"]
		names = ["fireplace", "living_room"]
		for name in names:	
			for mvt_num in range(0,11):
				im2data(name, mvt_num, aff=False)

