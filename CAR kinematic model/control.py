import numpy as np
from scipy.optimize import minimize
import copy


class Car_Dynamics:
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt):
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.x = x_0
        self.y = y_0
        self.v = v_0
        self.psi = psi_0
        self.state = np.array([[self.x, self.y, self.v, self.psi]]).T

    def move(self, accelerate, delta):
        x_dot = self.v*np.cos(self.psi)
        y_dot = self.v*np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)/self.L
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T

    def update_state(self, state_dot):
        # self.u_k = command
        # self.z_k = state
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.v = self.state[2,0]
        self.psi = self.state[3,0]


# create classes for diWheel robot and truck robot
class Trailer_Dynamics(Car_Dynamics):
    # equations are sourced from MATLAB article: https://www.mathworks.com/help/nav/ug/reverse-capable-motion-planning-for-tractor-trailer-model-using-plannercontrolrrt.html
    # and laValle: https://msl.cs.uiuc.edu/planning/node661.html
    def __init__(self, x_0, y_0, v_0, psi_0, vehicle_length, dt,trailer_length,beta):
        super().__init__(x_0, y_0, v_0, psi_0, vehicle_length, dt)                  # inherit some of the variables from the Car class
        self.trailer_L = trailer_length
        # self.theta = theta              # leading car orientation
        self.beta = beta                # trailer orientation
        self.v_trailer=v_0
        self.x_trailer = 0
        self.y_trailer = 0
        self.state = np.array([[self.x, self.y, self.v, self.psi,self.x_trailer,self.y_trailer,self.v_trailer,self.beta]]).T

    def move(self, accelerate, delta):
        x_dot = self.v*np.cos(self.psi)
        y_dot = self.v*np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)/self.L
        x_dot_trailer = self.v_trailer*np.cos(self.psi + self.beta)
        y_dot_trailer = self.v_trailer * np.sin(self.psi + self.beta)
        v_dot_trailer = 0
        beta_dot = self.v_trailer*np.tan(delta)/self.trailer_L

        return np.array([[x_dot, y_dot, v_dot, psi_dot,x_dot_trailer,y_dot_trailer,v_dot_trailer,beta_dot]]).T

    def update_state(self, state_dot):
        # self.u_k = command
        # self.z_k = state
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.v = self.state[2,0]
        self.psi = self.state[3,0]
        self.x_trailer = self.state[4,0]
        self.y_trailer = self.state[5,0]
        self.v_trailer = self.state[6,0]
        self.beta = self.state[7,0]
class DiWheel_Dyanmics(Car_Dynamics):
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt, theta_0,wheel_base = 2):
        super().__init__(x_0, y_0, v_0, psi_0, length, dt)
        self.theta = theta_0
        self.wheel_base = wheel_base        # need to specify wheel_base later
        self.state = np.array([[self.x, self.y, self.v, self.theta]]).T

    def move_diwheel(self, accelerate, delta):
        v_left = v_right = accelerate*delta
        # linear and angular velocities
        v = (v_left + v_right)/2.0
        omega = (v_right-v_left)/self.wheel_base
        x_dot = v * np.cos(self.theta)
        y_dot = v * np.sin(self.theta)
        v_dot = accelerate
        self.theta += omega * delta
        self.theta %= (2 * np.pi)
        # theta returns the orientation of the vehicle
        return np.array([[x_dot, y_dot, v_dot, self.theta]]).T

    def update_state(self, state_dot):
        # self.u_k = command
        # self.z_k = state
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.v = self.state[2,0]
        self.theta = self.state[3,0]

class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
    
        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i])
            mpc_car.update_state(state_dot)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]

