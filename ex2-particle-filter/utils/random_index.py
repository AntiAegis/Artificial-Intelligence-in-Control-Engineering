#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
np.random.seed(2)


#------------------------------------------------------------------------------
#	Function outputs a random index based on a given distribution
#------------------------------------------------------------------------------
def random_index(W):
	"""
	[Arguments]
		W : (list) List of important factors.

	[Return]
		idx : (int) Random index based on the given distribution of W.
	"""
	n_particles = len(W)
	total = np.sum(W)
	p = W/total
	return np.random.choice(n_particles, (), p=p)


#------------------------------------------------------------------------------
#	Test bench to verify the written function
#------------------------------------------------------------------------------
# x = [10,1,3,4,5,5,5,5,5,6,6,7,7,7,3,9] 
# total = np.sum(x) 
# p1 = x/total

# y = []
# for i in range(100000):
# 	y.append(random_index(x))

# _, count = np.unique(y, return_counts=True)
# total = np.sum(count) 
# p2 = count/total

# for i in range(len(p1)):
# 	print("%.4f\t%.4f" % (p1[i], p2[i]))