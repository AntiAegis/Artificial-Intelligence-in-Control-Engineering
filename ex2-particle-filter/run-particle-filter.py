#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
from scipy.io import loadmat
from utils.particle_filter import ParticleFilter
from time import time
start = time()


#------------------------------------------------------------------------------
#	Parameters
#------------------------------------------------------------------------------
n_particles = 100
sigma_v, sigma_g = 0.5, 3/180*np.pi
sigma_r, sigma_b = 0.2, 2/180*np.pi
wb = 4
time_step = 0.025
particle = "max"

#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Load data
data = loadmat("data20171107.mat")
landmarks, X_gt, Z, U, X_ODO = data["lm"], data["XTRUE"], data["Z"], data["VG"], data["XODO"]
n_steps = X_gt.shape[1]

# Create a Particle Filter instance
particle_filt = ParticleFilter(
	n_particles=n_particles,
	n_steps=n_steps,
	landmarks=landmarks,
	sigma_v=sigma_v,
	sigma_g=sigma_g,
	sigma_r=sigma_r,
	sigma_b=sigma_b,
	wb=wb,
	time_step=time_step,
)

# Perform loops
x_start = X_gt[:, 0, np.newaxis]
X_record, W_record = particle_filt.loop_over_steps(x_start, U, Z)

# Visualize the result
mse = particle_filt.compute_MSE(X_gt, X_record, W_record, particle)
print("MSE: %.6f" % (mse))
print("Time:", time()-start)
particle_filt.visualize(X_gt, X_ODO, X_record, W_record, particle)