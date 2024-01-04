from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import cv2
import scipy.optimize as optimize

import Useful_functions as uf

np.set_printoptions(precision = 3, suppress=True)

def horizontal_stereo(m, calib_param):
	F, D, O, f, s, t, pix = calib_param
	
	P_hat = []
	P_hat_indexs = []
	
	D_ = np.array([	[1, 0, 0, O[0]],
					[0, 1, 0, O[1]],
					[0, 0, 1, D],
					[0, 0, 0, 1]])
	Kc = np.array([	[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, -1/F, 1]])
							
	#estimate the depth for each point, for each line
	for k in np.unique(m[:,4]):
		m_k = m[np.where(m[:,4]==k)]
		lines_visible = np.unique(m_k[:,3])
		P_j_hat = []
		P_j_hat_ = []
		for j in lines_visible:
			j = int(j)
			m_j_k = m_k[np.where(m_k[:,3]==j)]
			column_visible = np.unique(m_j_k[:,2])
			
			depths = []
			
			for k1 in range(column_visible.shape[0]-1):
				i1 = int(column_visible[k1])
				for k2 in range(column_visible.shape[0]-k1-1):
					i2 = int(column_visible[k1+k2+1])
					m_i1_j_k = m_j_k[np.where(m_j_k[:,2]==i1)]
					m_i2_j_k = m_j_k[np.where(m_j_k[:,2]==i2)]

					dist_ = s[i2]-s[i1]
					
					disparity_ = m_i1_j_k[0,0]-m_i2_j_k[0,0]
					if np.linalg.norm(disparity_)>0.00001:
						depths.append(-f*dist_/disparity_)
					else:
						depths.append((-f*dist_/0.00001)*np.sign(disparity_))
			
			if len(depths)>0:
				depth = np.median(depths)
		
				pos_x = []
				pos_y = []
				#estimate the 3D points
				for i in column_visible:
					i = int(i)
					m_i_j_k = m_j_k[np.where(m_j_k[:,2]==i)]
					pos_x.append(((depth*m_i_j_k[0,0])/f)+s[i])
					pos_y.append(((depth*m_i_j_k[0,1])/f)+t[j])
				
				P_j_hat_.append(np.array([j, np.median(pos_x), np.median(pos_y), depth, 1, k]))
				#reproject through main lens
				D_Kc_inv = np.linalg.inv(np.matmul(D_,Kc))		
				
				for P_j_h_ in P_j_hat_:
					P_j_h = np.matmul(D_Kc_inv,P_j_h_[1:-1])
					#projection function	
					P_j_hat.append(np.array([j, P_j_h[0]/P_j_h[-1], P_j_h[1]/P_j_h[-1], P_j_h[2]/P_j_h[-1], 1, k]))
				
		if len(P_j_hat)>3:
			P_hat.append(np.array([np.median(np.array(P_j_hat)[:,1]), np.median(np.array(P_j_hat)[:,2]), np.median(np.array(P_j_hat)[:,3]), 1]))
			P_hat_indexs.append(k)

	P_hat = np.array(P_hat)
	P_hat_indexs = np.array(P_hat_indexs)
	return P_hat, P_hat_indexs

if __name__ == "__main__":
	#load intrisic parameters
	calib_param = uf.load_calib_param()
	F, D, O, f, s, t, pix = calib_param
	
	name = "rabbit"
	data_folrder = "Data/Data_"+name+"_Mvt_1"
	
	#load rslf image
	rslf_image = uf.load_rslf_image(data_folrder)
	
	#estimate the structure for each line t
	P_hat, _ = horizontal_stereo(rslf_image, calib_param)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(P_hat[:,0],P_hat[:,1],P_hat[:,2], c='red', label='P_hat')
	ax.set_xlabel('$X$', fontsize=10, rotation = 0)
	ax.set_ylabel('$Y$', fontsize=10, rotation = 0)
	ax.set_zlabel('$Z$', fontsize=10, rotation = 0)
	ax.legend()
	uf.set_axes_equal(ax)
	plt.show()
