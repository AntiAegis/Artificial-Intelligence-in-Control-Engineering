#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
from scipy.io import loadmat


#------------------------------------------------------------------------------
#	Test bench to verify the written function
#------------------------------------------------------------------------------
# Load data from file mat
data = loadmat("data20171107.mat")
lm, XTRUE, XODO, Z, VG = data["lm"], data["XTRUE"], data["XODO"], data["Z"], data["VG"]