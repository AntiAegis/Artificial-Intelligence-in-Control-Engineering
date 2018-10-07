#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt

from utils.process_model import process_model
from utils.important_factor import important_factor
from utils.random_index import random_index


#------------------------------------------------------------------------------
#	Class of Particle Filter
#------------------------------------------------------------------------------
class ParticleFilter(object):
	"""
	[Arguments]
		n_particles : (int) Number of particles.

		n_steps : (int) Number of time steps.

		landmarks : (ndarray) Position matrix of landmarks. The shape of lm is
			(2, n_steps).

		sigma_v : (float) Standard deviation of the velocity noise.

		sigma_g : (float) Standard deviation of the stearing angle noise.

		sigma_r : (float) Standard deviation of the range noise of the lazer.

		sigma_b : (float) Standard deviation of the bearing angle noise
			(radian) of the lazer.

		wb : (float) 

		time_step : (float) The interval between consecutive controls.
	"""
	# Initialization
	def __init__(self, n_particles, n_steps, landmarks,
					sigma_v, sigma_g, sigma_r, sigma_b,
					wb, time_step):
		super(ParticleFilter, self).__init__()
		self.n_particles = n_particles
		self.n_steps = n_steps
		self.landmarks = landmarks
		self.sigma_v = sigma_v
		self.sigma_g = sigma_g
		self.sigma_r = sigma_r
		self.sigma_b = sigma_b
		self.wb = wb
		self.time_step = time_step


	# Prediction phase in a time step
	def predict_in_step(self, X, u, z):
		"""
		[Arguments]
			X : (list) List of current position particles.

			u : (ndarray) Control signal, including velocity and stearing angle
				(radian). The shape of u is (2, 1).

			z : (ndarray) Ground truth of measurement, as described in the
				readme file. The shape of z is (3, 5).

		[Return]
			X_bar : (list) List of predicted position particles.

			W : (list) List of important factor particles.
		"""
		X_bar, W = [], []

		for m in range(self.n_particles):
			x_pred = process_model(
				x=X[m],
				u=u,
				sigma_v=self.sigma_v,
				sigma_g=self.sigma_g,
				wb=self.wb,
				time_step=self.time_step,
			)
			w = important_factor(
				x=x_pred,
				z=z,
				lm=self.landmarks,
				sigma_r=self.sigma_r,
				sigma_b=self.sigma_b,
			)
			X_bar.append(x_pred)
			if w is not None:
				W.append(w)

		return X_bar, W


	# Correction phase in a time step
	def correct_in_step(self, X_bar, W):
		"""
		[Arguments]
			X_bar : (list) List of predicted position particles.

			W : (list) List of important factor particles.

		[Return]
			X : (list) New position particles.
		"""
		X = []

		for m in range(self.n_particles):
			idx = random_index(W)
			X.append(X_bar[idx])

		return X


	# Perform loop over time steps
	def loop_over_steps(self, x_start, U, Z):
		"""
		[Arguments]
			x_start : (ndarray) Starting position, including x coordinate, y
				coordinate, and angle (radian). The shape of x_start is (3, 1).

			U : (ndarray) Control signal matrix as described in the readme file.
				The shape of U is (2, N).

			Z : (ndarray) Ground truth matrix of measurement, as described in
			the readme file. The shape of Z is (3, 5, N).

		[Return]
			X_record : (list) List of position particles over time steps.

			W_record : (list) List of important factors over time steps.
		"""
		X = [x_start] * self.n_particles
		X_record = [X]
		W_record = [[1.0/self.n_particles] * self.n_particles]

		for n in range(1, self.n_steps):
			u = U[:, n, np.newaxis]
			z = Z[:, :, n]

			X_bar, W = self.predict_in_step(X, u, z)
			if len(W)==0:
				W = W_record[-1]
			X = self.correct_in_step(X_bar, W)

			X_record.append(X)
			W_record.append(W)

		return X_record, W_record


	# Compute Mean Square Error
	def compute_MSE(self, X_true, X_record, W_record=None, particle="max"):
		"""
		[Arguments]
			X_true : (ndarray) Ground truth matrix of positions over time
				steps. The shape of X_true is (3, N).

			X_record : (list) List of position particles over time steps.

			W_record : (list) List of important factors over time steps.

			particle : (str) Indicate which particle's result is visualized.
				particle must be "max", "min", or "median".
		"""
		x_true, y_true = X_true[0,:], X_true[1,:]
		x_odo, y_odo = [], []

		# Calculate X_ODO
		if W_record is not None:
			for n in range(self.n_steps):
				X, W = X_record[n], W_record[n]

				if particle=="max":
					idx = int(W.index(np.max(W)))
				elif particle=="min":
					idx = int(W.index(np.min(W)))
				elif particle=="median":
					idx = int(np.argmin(np.abs(W-np.median(W))))

				x_odo.append(X[idx][0, 0])
				y_odo.append(X[idx][1, 0])

			x_odo, y_odo = np.array(x_odo), np.array(y_odo)

		else:
			x_odo, y_odo = X_record[0, :], X_record[1, :]

		# Compute MSE
		mse = 0.5 * np.sum((x_odo-x_true)**2 + (y_odo-y_true)**2)
		return mse


	# Visualize the progress
	def visualize(self, X_true, X_ODO, X_record, W_record, particle="max"):
		"""
		[Arguments]
			X_true : (ndarray) Ground truth matrix of positions over time
				steps. The shape of X_true is (3, N).

			X_record : (list) List of position particles over time steps.

			W_record : (list) List of important factors over time steps.

			particle : (str) Indicate which particle's result is visualized.
				particle must be "max", "min", or "median".
		"""
		x_true, y_true = X_true[0,:], X_true[1,:]
		x_odo_true, y_odo_true = X_ODO[0,:], X_ODO[1,:] 
		x_lm, y_lm = self.landmarks[0,:], self.landmarks[1,:]
		x_odo, y_odo = [], []

		for n in range(self.n_steps):
			X, W = X_record[n], W_record[n]

			if particle=="max":
				idx = int(W.index(np.max(W)))
			elif particle=="min":
				idx = int(W.index(np.min(W)))
			elif particle=="median":
				idx = int(np.argmin(np.abs(W-np.median(W))))

			x_odo.append(X[idx][0, 0])
			y_odo.append(X[idx][1, 0])

		plt.figure(1)
		plt.plot(x_true, y_true, "-r")
		plt.plot(x_odo_true, y_odo_true, "-g")
		plt.plot(x_odo, y_odo, "--b")
		plt.plot(x_lm, y_lm, "xm")
		plt.plot(x_true[0], y_true[0], "o")
		plt.legend(["Ground truth", "XODO", "Prediction", "Landmark", "Starting"])
		plt.show()