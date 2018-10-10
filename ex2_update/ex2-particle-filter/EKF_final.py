#   Libraries
#------------------------------------------------------------------------------
from scipy.io import loadmat
from math import cos, sin, sqrt, atan2, tan
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, dot
from numpy.random import randn
from EKF import ExtendedKalmanFilter as EKF

np.random.seed(2)
#------------------------------------------------------------------------------
#   Parameters
#------------------------------------------------------------------------------
sigma_v, sigma_g = 0.5, 3/180*np.pi
sigma_r, sigma_b = 0.2, 2/180*np.pi
wb = 4
time_step = 0.025

#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------
# Load data
data = loadmat("data20171107.mat")
landmarks, X_gt, Z, U, XODO = data["lm"], data["XTRUE"], data["Z"], data["VG"], data["XODO"]
n_steps = X_gt.shape[1]

x_true, y_true, phi_true = X_gt[0,:], X_gt[1,:], X_gt[2,:]
x_odo, y_odo, phi_odo = XODO[0,:], XODO[1,:], XODO[2,:]
x_lm, y_lm = landmarks[0,:], landmarks[1,:]


def residual(a,b):
    # Bearing angle is normalized to [-pi, pi)
    if a[1] >= b[1]:
        y = a - b
    else:
        y = b - a
    y[1] = y[1] % (2*np.pi)
    if y[1] > np.pi:
        y[1] -= 2*np.pi
    return y


class RobotEKF(EKF):
    def __init__(self, dt, sigma_v, sigma_g, wheelbase = 4):
        EKF.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.sigma_v = sigma_v
        self.sigma_g = sigma_g

    def predict(self, x, u, dt):
        v = u[0] + self.sigma_v * np.random.randn()
        theta = u[1] + self.sigma_g * np.random.randn()

        dist = v*dt
        phi = x[2]
        b = dist / self.wheelbase * sin(theta)
        
        sinhb = sin(theta + phi + b)
        coshb = cos(theta + phi + b)

        return x + array([[dist*coshb],
                          [dist*sinhb],
                          [b]])

def H_of(x, p):
    ''' Compute Jacobian of H matrix where h(x) computes the range and
    bearing to a landmark for state x '''

    px = p[0]
    py = p[1]
    hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
    dist = np.sqrt(hyp)

    H = array(
        [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
         [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1]])
    return H


def Hx(x, p):
    ''' Takes a state variable and returns the measurement that would
    correspond to that state.
    '''
    px = p[0]
    py = p[1]
    dist = np.sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

    Hx = array([[dist],
                [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
    return Hx

def calculate_MSE(x_TRUE, x_predict, y_predict):
    x_true, y_true = x_TRUE[0,:], x_TRUE[1,:]
    MSE= 0.5 * np.sum((x_predict-x_true)**2 + (y_predict-y_true)**2)
    return MSE


dt = 0.025
ekf = RobotEKF(dt, wheelbase=4, sigma_v=sigma_v, sigma_g=sigma_g)
ekf.x = X_gt[:, 0, np.newaxis]

sigma_steer =  np.radians(2)

ekf.P = np.diag([1., 1., 1.])
ekf.R = np.diag([sigma_r**2, sigma_b**2])

xp = ekf.x.copy()
xp_0= np.array([])
xp_1= np.array([])

for k in range(len(U[0])):
    xp = ekf.predict(xp, U[:,k], dt)       # Simulate robot   
    xp_0= np.append(xp_0,xp[0])
    xp_1= np.append(xp_1,xp[1])

    if k % 2 == 0:
        for x, y in zip(landmarks[0,:],landmarks[1,:]):
            d = sqrt((x - xp[0, 0])**2 + (y - xp[1, 0])**2)  + randn()*sigma_r
            a = atan2(y - xp[1, 0], x - xp[0, 0]) - xp[2, 0] + randn()*sigma_b
            z = np.array([[d], [a]])

            ekf.update(z, HJacobian=H_of, Hx=Hx, residual=residual,
                    args=([x,y]), hx_args=([x,y]))

# Visualize the result

MSE= calculate_MSE(X_gt, xp_0, xp_1)
print("MSE= %.3f" % (MSE))

plt.plot(x_true, y_true, "-r")
plt.plot(x_odo, y_odo, "b")
plt.plot(x_lm, y_lm, "xm")
plt.plot(xp_0, xp_1, "--g")
plt.legend(["XTRUE","XODO","landmarks","Prediction"])
plt.show()
