from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import csv
import os

import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(' Using device: {0}'.format(device))

#import my functions
import static_res as gs
import Useful_functions as uf

np.set_printoptions(precision = 3, suppress=True)
torch.set_printoptions(precision = 3)

def sig(x):
 return 1/(1 + torch.exp(-x))
 
def np2torch(np_var):
	np_var_np = np_var.astype(np.float32)
	if np_var.shape == ():
		np_var_np = np.array([np_var_np])
	return torch.from_numpy(np_var_np).to(device)

def sep_t_a_Omega(a_Omega):
	threshold = 0.00000001
	Omega_ = torch.linalg.norm(a_Omega)
	if Omega_ >threshold:
		a_ = a_Omega/Omega_
	else:
		a_ = torch.tensor([[0.],[0.],[1.]]).to(device)
	return a_, Omega_

def init_param(rslf_image, calib_param):
	#estimate the structure
	P_hat, P_hat_indexs = gs.horizontal_stereo(rslf_image, calib_param)
	
	#Clean some outliers from rslf_image (removed from horizontal stereo):
	cleaned_rslf_image = []
	new_k = 0
	for k in P_hat_indexs:
		for pnt in rslf_image[np.where(rslf_image[:,4]==k)]:
			new_pnt = np.copy(pnt)
			new_pnt[4] = new_k
			cleaned_rslf_image.append(new_pnt)
		new_k+=1
	cleaned_rslf_image = np.array(cleaned_rslf_image)
	
	#Convert data to torch
	rslf_image = np2torch(cleaned_rslf_image)
	P_init = np2torch(P_hat)
	
	a_Omega_init = np2torch(np.array([[0.01],[0.01],[0.01]]))
	V_init = np2torch(np.array([[0.],[0.],[0.]]))
	
	F, D, O, f, s, t, pix = calib_param
	F = np2torch(F)
	D = np2torch(D)
	O = np2torch(O)
	f = np2torch(f)
	s = np2torch(s)
	t = np2torch(t)
	pix = np2torch(pix)
	calib_param = F, D, O, f, s, t, pix
	
	return rslf_image, P_init, a_Omega_init, V_init, calib_param
	
def loss_torch(rslf_image, P_opti_n, a_Omega, V_opti, calib_param, CoM_opti, scale, K):
	F, D, O, f, s, t, pix = calib_param
	K_st, t_, s_, t_stat = K
	
	a, Omega = sep_t_a_Omega(a_Omega)
	
	aa = torch.tensor([[0,-a[2][0],a[1][0]],[a[2][0],0,-a[0][0]],[-a[1][0],a[0][0],0]]).to(device)
	deltaR_t = torch.repeat_interleave(torch.matmul(a,a.T)[:,:,None], t_.shape[0], dim=2)*(1-torch.cos(t_stat*Omega)) + torch.repeat_interleave(torch.eye(3).to(device)[:,:,None], t_.shape[0], dim=2)*torch.cos(t_stat*Omega) + torch.repeat_interleave(aa[:,:,None], t_.shape[0], dim=2)*torch.sin(t_stat*Omega)
	deltaT_t = t_stat*V_opti
	RT = torch.cat((deltaR_t, deltaT_t[:,None,:]), axis=1)
	RT = torch.transpose(RT, 0, 2)
	RT = torch.transpose(RT, 1, 2)
	
	P_opti_n_st = P_opti_n[rslf_image[:,4].type(torch.int64)]
	P_n = torch.matmul(RT, P_opti_n_st[:,:,None])
	P_n1 = torch.clone(P_n)
	
	# Denormalization
	P_n = P_n1*scale + torch.transpose(CoM_opti, 0, 1)
	P_n = torch.cat((P_n, torch.repeat_interleave(torch.tensor([1]).to(device)[:,None, None], t_.shape[0], dim=0)), axis=1)
	
	m = torch.matmul(K_st, P_n)
	
	ui_hat = (-m[:,0,0]/m[:,2,0])
	vi_hat = (-m[:,1,0]/m[:,2,0])
	
	ui_bar = rslf_image[:,0]
	vi_bar = rslf_image[:,1]
	
	loss = torch.linalg.norm(sig(abs(ui_hat-ui_bar))-0.5) + torch.linalg.norm(sig(abs(vi_hat-vi_bar))-0.5)
	
	return loss
	
def opti_reproj_err(rslf_image, init_params, hyp_param):
	iterations, lr = hyp_param
	rslf_image, P_init, a_Omega_init, V_init, calib_param = init_params
	
	#Normalization of P_init: Center of Mass and scale
	CoM_init = torch.tensor([[torch.mean(P_init[:,0]),torch.mean(P_init[:,1]),torch.mean(P_init[:,2])]]).to(device)
	P_init_n = torch.clone(P_init)
	P_init_n[:,:3] = P_init[:,:3]-CoM_init
	scale = torch.max(P_init_n)
	P_init_n = P_init_n/scale
	P_init_n[:,3]=1
	
	#Construct a graph for the variables to optimize
	P_opti_n = P_init_n.clone().detach().requires_grad_(True)
	a_Omega_opti = a_Omega_init.clone().detach().requires_grad_(True)
	V_opti = V_init.clone().detach().requires_grad_(True)
	CoM_opti = CoM_init.clone().detach().requires_grad_(True)
	
	list_loss = []
	#Set optimizer
	opt = optim.Adam([a_Omega_opti, V_opti, P_opti_n, CoM_init], lr=lr)
	
	print("\n Reprojection error minimization")
	time_zero = time.time()
	F, D, O, f, s, t, pix = calib_param
	
	t_ = t[rslf_image[:,3].type(torch.int64)]
	s_ = s[rslf_image[:,2].type(torch.int64)]
	
	## We want the static position to be the central position:
	t_stat = (t_-t[-2].item()/2)
	
	s_1_3 = torch.repeat_interleave(torch.tensor([[0, 0, 1, 0],[0, 0, 0, 0],[0, 0, 0, 0]]).to(device)[:,:,None], s_.shape[0], dim=2)
	s_1_4 = torch.repeat_interleave(torch.tensor([[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0]]).to(device)[:,:,None], s_.shape[0], dim=2)
	
	t_2_3 = torch.repeat_interleave(torch.tensor([[0, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 0]]).to(device)[:,:,None], t_.shape[0], dim=2)
	t_2_4 = torch.repeat_interleave(torch.tensor([[0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 0, 0]]).to(device)[:,:,None], t_.shape[0], dim=2)
	
	mat_s = -(f/F)*(s_1_3*O[0]-s_1_3*s_) + f*(s_1_4*O[0]-s_1_4*s_)
	
	mat_t = -(f/F)*(t_2_3*O[1]-t_2_3*t_) + f*(t_2_4*O[1]-t_2_4*t_)
	
	mat_other_s = torch.repeat_interleave(torch.tensor([[f, 0, 0, 0],[0, f, 0, 0],[0, 0, 1-(D/F), D]]).to(device)[:,:,None], s_.shape[0], dim=2)
	
	K_st = mat_s + mat_other_s + mat_t
	K_st = torch.transpose(K_st, 0, 2)
	K_st = torch.transpose(K_st, 1, 2)
	
	K = K_st, t_, s_, t_stat
	
	for iteration in range(iterations):
		# perform the optimization
		
		opt.zero_grad()
		loss = loss_torch(rslf_image, P_opti_n, a_Omega_opti, V_opti, calib_param, CoM_opti, scale, K)
		loss.backward(retain_graph=True)
		opt.step()
		list_loss.append(loss.detach().cpu().numpy())
		
		if iteration//(iterations//10) == iteration/(iterations//10) and iteration!=0:
			print(' ', iteration,'/', iterations,':',loss.detach().cpu().numpy(), '       ')
		if iteration//50 == iteration/50 and iteration!=0:
			print(' Progress: '+str(int((iteration/iterations)*100))+'% ('+str(int(((iterations/iteration)-1)*(time.time()-time_zero)))+"sec remaining)        ", end='\r')
	
	print(" Progress: Finished (after "+str(int((time.time()-time_zero)))+"sec)")
	
	### DENORMALIZATION
	P_opti_n = P_opti_n*scale
	P_opti = torch.clone(P_opti_n)
	P_opti[:,:3] = P_opti_n[:,:3]+CoM_opti
	V_opti = V_opti*scale
	
	a_opti, Omega_opti = sep_t_a_Omega(a_Omega_opti)
	
	#Convert back to numpy
	P_opti_ = P_opti.detach().cpu().numpy()
	P_opti_[:,3] = np.ones((P_opti_.shape[0],))
	
	a_opti = a_opti.detach().cpu().numpy()
	Omega_opti = Omega_opti.detach().cpu().numpy()
	V_opti = V_opti.detach().cpu().numpy()
	return P_opti_, a_opti, Omega_opti, V_opti, list_loss
	
def	nlmethod(name, hyp_param):
	data_folder = "Data/Data_"+name
	
	#load intrisic parameters
	calib_param = uf.load_calib_param()
	F, D, O, f, s, t, pix = calib_param
	
	#load rslf image
	rslf_image = uf.load_rslf_image(data_folder)
	
	#initilization
	init_params = init_param(rslf_image, calib_param)
	
	#minimize reprojection error
	P_opti, a_opti, Omega_opti, V_opti, list_loss = opti_reproj_err(rslf_image, init_params, hyp_param)
	
	#print and plot
	print('\n  a=',a_opti.T,'\n  Omega=',Omega_opti, '\n  V=',V_opti.T)
	
	#write the coordinates of the estimated points in a csv file
	with open('Data/Data_'+ name +'/result.csv', 'w') as f_:
		writer = csv.writer(f_)
		for i in range(P_opti.shape[0]):
			writer.writerows(np.array([P_opti[i, :]]))
					
	#write the estimated movement in a csv file
	with open('Data/Data_'+ name +'/mov_result.csv', 'w') as f_:
		writer = csv.writer(f_)
		writer.writerows(a_opti)
		writer.writerows(np.array([[Omega_opti]]))
		writer.writerows(V_opti)
	
	return P_opti, a_opti, Omega_opti, V_opti, list_loss


def plot_result(res):
	P, a, Omega, V, list_loss = res

	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(P[:,0],P[:,1],P[:,2], c='black', label='P_opti', marker='.', alpha=1)

	uf.set_axes_equal(ax)
	ax.set_xlabel('$X$', fontsize=10, rotation = 0)
	ax.set_ylabel('$Y$', fontsize=10, rotation = 0)
	ax.set_zlabel('$Z$', fontsize=10, rotation = 0)
	ax.view_init(elev=-75, azim=90)
	plt.show()
	
	plt.figure()
	plt.plot(list_loss)
	plt.show()
	return 0

if __name__ == "__main__":
	#--- HYPER PARAMETERS --------------------------------------------
	iterations = 10000
	lr = 0.001
	#-----------------------------------------------------------------
	
	hyp_param = iterations, lr
	
	if 1:
		#data folder
		nam = "rabbit"
		mvt_num = 10
		
		name = nam + "_Mvt_" + str(mvt_num)
		print("\n Scene", name)
		res = nlmethod(name, hyp_param)
		
		plot_result(res)

	else:
		names = ["chart", "rabbit", "table", "bedroom", "couch", "fireplace", "living_room"]
		for nam in names:
			for mvt_num in range(0,11):
				name = nam + "_Mvt_" + str(mvt_num)
				print("\n Scene", name)
				res = nlmethod(name, hyp_param)
				if 0:
					plot_result(res)
				

