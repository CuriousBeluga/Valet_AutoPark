import cv2
import numpy as np

class Environment():
    def __init__(self,obstacles):

        self.margin = 5

        #coordinates are in [x,y] format
        self.car_length = 80
        self.car_width = 40
        self.wheel_length = 15
        self.wheel_width = 7
        self.wheel_positions = np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
        
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)



        #height and width
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3))
        self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255
        self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255
        self.place_obstacles(obstacles)
                
    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        for ob in obstacles:
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=0
    
    def draw_path(self, path):
        path = np.array(path)*10
        color = np.random.rand(3)
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, phi, delta):
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=phi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=phi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+phi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=phi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered
