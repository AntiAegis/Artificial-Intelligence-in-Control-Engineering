#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#	Function represents the process model
#------------------------------------------------------------------------------
def process_model(x,u, sigma_v, sigma_g , wb, time_step):
	"""
	[Arguments]
		x : (ndarray) Current position, including x coordinate, y coordinate,
			and angle (radian). The shape of x is (3, 1).

		u : (ndarray) Control signal, including velocity and stearing angle
			(radian). The shape of u is (2, 1).

		sigma_v : (float) Standard deviation of the velocity noise.

		sigma_g : (float) Standard deviation of the stearing angle noise.

	[Return]
		x_pred : (ndarray) Predicted position, including x coordinate, y
			coordinate and angle (radian). The shape of x_pred is (3, 1).
	"""
	# calculate V* = V + delta(V)
    new_V = u[0] + sigma_v * np.random.randn()
    
    # calculate G* = G + delta(G)
    new_G = u[1] + sigma_g * np.random.randn()
    
    # calculate new angle 
    new_angle = x[2] + new_V * time_step * np.sin(new_G)/ wb
    
    # calculate y
    new_y = x[1]+ new_V * time_step * np.sin(new_G + new_angle)
    
    # calculate x
    new_x = x[0] + new_V * time_step * np.cos(new_G + new_angle)
    
    # result
    new_coordinates=np.array([[new_x],[new_y],[new_angle]])
    return new_coordinates


#------------------------------------------------------------------------------
#	Test bench to verify the written function
#------------------------------------------------------------------------------
x= np.array([1,1,1]) # current location
u= [2,3] # control signal
a= process_model(x,u, 0.5,0.5,4,1)
print(a) # result
