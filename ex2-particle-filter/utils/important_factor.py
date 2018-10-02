#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np 


#------------------------------------------------------------------------------
#	Function computes the important factor
#------------------------------------------------------------------------------
def distribution(z_pred, z_gt, sigma_r, sigma_b):
	def gaussian(x, mu, sigma):
		return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*(((x-mu)/sigma)**2))

	dz = z_pred - z_gt
	dr, db = dz[0,0], dz[1,0]

	pr = gaussian(dr, 0, sigma_r)
	pb = gaussian(db, 0, sigma_b)
	return pr*pb


def important_factor(x, z, lm, sigma_r, sigma_b):
	"""
	[Arguments]
		x : (ndarray) Current position, including x coordinate, y coordinate,
			and angle (radian). The shape of x is (3, 1).

		z : (ndarray) Ground truth of measurement, as described in the readme
			file. The shape of z is (3, 5).

		lm : (ndarray) Position matrix of landmarks.

		sigma_r : (float) Standard deviation of the range noise of the lazer.

		sigma_b : (float) Standard deviation of the bearing angle noise
			(radian) of the lazer.

	[Return]
		w : (float) Important factor.
	"""
	importance_factor = []

	for i in range(z.shape[1]):
		if not np.isnan(z[2, i]):
			id_lm = int(z[2, i] - 1) 
			xL, yL = lm[0, id_lm], lm[1, id_lm]
			xt, yt, phit = x[0], x[1], x[2]

			r = np.sqrt((xt-xL)**2 + (yt-yL)**2)
			b = np.arctan((yt-yL)/(xt-xL)) - phit
			# b = np.arctan2(yt-yL, xt-xL) - phit

			z_pred = np.array([r, b])
			zt = np.array([[z[0,i]], [z[1,i]]])

			w = distribution(z_pred, zt, sigma_r, sigma_b)
			importance_factor.append(w)

	w = None if importance_factor==[] else np.mean(importance_factor)
	return w


#------------------------------------------------------------------------------
#	Test bench to verify the written function
#------------------------------------------------------------------------------
# from scipy.io import loadmat

# data = loadmat("../data20171107.mat") 
# lm, z, x = data["lm"], data["Z"], data["XODO"]
# z = z[:,:,31]
# x = x[:,31] 
# w = important_factor(x, z, lm, 0.2, 2) 
# print(w)