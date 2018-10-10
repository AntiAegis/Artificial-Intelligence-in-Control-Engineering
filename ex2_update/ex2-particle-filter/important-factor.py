#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np 
from scipy.io import loadmat

#------------------------------------------------------------------------------
#	Function represents the process model
#------------------------------------------------------------------------------
def distribution(x, z, sigma_r, sigma_b):
	cov_matrix = np.array([[pow(sigma_r, 2), 0], [0, pow(sigma_b, 2)]])
	det = np.linalg.det(cov_matrix) 
	inv_cov_matrix = np.linalg.inv(cov_matrix)  
	part_1 = 1/(2*np.pi*np.sqrt(det)) 
	part_2 = np.matmul(np.transpose((x - z)), inv_cov_matrix) 
	part_3 = np.matmul(part_2, (x - z)) 
	return part_1*np.exp(- part_3)[0][0] 

def compute_important_factor(x, z, lm, sigma_r, sigma_b):
	"""
	[Arguments]
		x : (ndarray) Current position, including x coordinate, y coordinate,
			and angle (radian). The shape of x is (3, 1).

		z : (ndarray) Ground truth of measurement, as described in the readme
			file. The shape of z is (3, 5).

		sigma_r : (float) Standard deviation of the range noise of the lazer.

		sigma_b : (float) Standard deviation of the bearing angle noise
			(radian) of the lazer.

	[Return]
		w : (float) Important factor.
	"""
	importance_factor = [] 
	count = 0 
	for i in range(5):
		if not np.isnan(z[2,i]):
			count = count + 1 
			id_lm = int(z[2,i] - 1) 
			xL = lm[0, id_lm]
			yL = lm[1, id_lm]
			xt = x[0] 
			yt = x[1] 
			phi = x[2] 
			r = np.sqrt(pow(xt - xL, 2) + pow(yt - yL, 2))  
			b = np.arctan((yt - yL)/(xt - xL) - phi) 
			z_pre = np.array([[r], [b]])
			zt = np.array([[z[0,i]], [z[1,i]]]) 
			w = distribution(z_pre, zt, sigma_r, sigma_b) 
			importance_factor.append(w) 
	total = np.sum(importance_factor) 
	importance_factor = total/count
	return importance_factor

#------------------------------------------------------------------------------
#	Test bench to verify the written function
#------------------------------------------------------------------------------
data = loadmat("data20171107.mat") 
lm = data["lm"] 
z = data["Z"]
z = z[:,:,31]
x = data["XODO"]
x = x[:,31] 
w = compute_important_factor(x, z, lm, 0.2, 2*np.pi/180) 
print(w) 