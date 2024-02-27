import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, Trailer_Dynamics, DiWheel_Dyanmics, MPC_Controller # Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # use this section to input start parameters
    # use an input function or something later
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=10, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=2, help='park position in parking1 out of 24') # choose parking spot with this
    args = parser.parse_args()
    logger = DataLogger()

    # initialize start and end poses
    start = np.array([args.x_start, args.y_start])
    end = np.array([args.x_end, args.y_end])

    # define obstacles
    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()



    # vehicle selection and environment creation
    flag = True
    while flag:
        vehicle = input("Please choose a vehicle, type the name without parentheses: (diwheel) (car) (trailer)\n")  # change for each type of vehicle
        if vehicle == "diwheel" or vehicle == "car" or vehicle == "trailer":
            flag = False
        else:
            print("I'm sorry, please type again")

    if vehicle == 'diwheel':
        car_length = 20  # diwheel
        car_width = 20  # diwheel
        wheel_length = 7  # diwheel
        wheel_width = 3  # diwheel
        length = 2
        dt = 0.2
        my_car = DiWheel_Dyanmics(start[0], start[1], 0, np.deg2rad(args.psi_start), length, dt, theta_0=0,
                                  wheel_base=2)  # diwheel
    elif vehicle == 'car':
        car_length = 80            #car
        car_width = 40            #car
        wheel_length = 15         #car
        wheel_width = 7           #car
        my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2) # regular car

    env = Environment(obs,vehicle,car_length,car_width,wheel_length, wheel_width)

    MPC_HORIZON = 5
    controller = MPC_Controller()
    if vehicle == "diwheel":
        res = env.render(my_car.x, my_car.y, my_car.theta, 0)  # diwheel
    elif vehicle == 'car':
        res = env.render(my_car.x, my_car.y, my_car.psi, 0)             # regular car

    cv2.imshow('environment', res)
    key = cv2.waitKey(1)


    # create path planning object
    park_path_planner = ParkPathPlanning(obs,vehicle)
    path_planner = PathPlanning(obs,vehicle)

    # planning parallel park
    new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]),vehicle)

    # creating path
    path = path_planner.plan_path(int(start[0]),int(start[1]),int(new_end[0]),int(new_end[1]))
    path = np.vstack([path, ensure_path1])

    # interpolating path
    interpolated_path = interpolate_path(path, sample_rate=5)
    interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])

    # env.draw_path(path)
    env.draw_path(interpolated_path)
    env.draw_path(interpolated_park_path)

    final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    # implement MPC controller
    print('driving to destination ...')

    for i in range(len(interpolated_path)):         # experimenting with render
        coord = interpolated_path[i]
        res = env.render(coord[0],coord[1],my_car.theta,1)

        cv2.imshow('environment', res)
    # animate final path
    for i,point in enumerate(final_path):
            # include if cases for different vehicle scenarios
            acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
            # my_car.update_state(my_car.move(acc,  delta))    # regular car
            my_car.update_state(my_car.move_diwheel(acc, delta))  # diwheel
            # res = env.render(my_car.x, my_car.y, my_car.psi, delta) # for regular car
            res = env.render(my_car.x, my_car.y, my_car.theta, delta)  # for diwheel
            logger.log(point, my_car, acc, delta)
            cv2.imshow('environment', res)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('res.png', res*255)


    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()

    cv2.destroyAllWindows()

