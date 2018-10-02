#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
from scipy.io import loadmat
from matplotlib import pyplot as plt


#------------------------------------------------------------------------------
#	Test bench to verify the written function
#------------------------------------------------------------------------------
# Load data from file mat
data = loadmat("data20171107.mat")
lm, XTRUE, XODO, Z, VG = data["lm"], data["XTRUE"], data["XODO"], data["Z"], data["VG"]


# Visualize data
x_true, y_true, phi_true = XTRUE[0,:], XTRUE[1,:], XTRUE[2,:]
x_odo, y_odo, phi_odo = XODO[0,:], XODO[1,:], XODO[2,:]
x_lm, y_lm = lm[0,:], lm[1,:]

for i in range(phi_odo.shape[0]):
	print("%.4f\t%.4f" % (phi_true[i], phi_odo[i]))

plt.figure(1)
plt.plot(x_true, y_true, "-r")
plt.plot(x_odo, y_odo, "--b")
plt.plot(x_lm, y_lm, "xm")
plt.legend(["Ground truth", "Prediction", "Landmark"])
plt.show()