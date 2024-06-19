"""
Plotting tool for 2D multi-robot system

author: Ashwin Bose (@atb033)
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_robot_and_obstacles(robot_state_history, obstacles, robot_radius, num_timesteps, sim_time, filename):
    fig, ax = plt.subplots()
    time = np.linspace(0, sim_time, num_timesteps)

    for i in range(num_timesteps):
        plt.cla()
        ax.set_xlim(-1, 15)
        ax.set_ylim(-1, 15)
        for obstacle in obstacles:
            circle = plt.Circle((obstacle[0, i], obstacle[1, i]), robot_radius, color='r')
            ax.add_artist(circle)

        circle = plt.Circle((robot_state_history[0, i], robot_state_history[1, i]), robot_radius, color='b')
        ax.add_artist(circle)

        plt.pause(0.01)

    plt.show()

def plot_robots_and_obstacles(robot_state_histories, robot_radius, num_timesteps, sim_time, filename):
    fig, ax = plt.subplots()
    time = np.linspace(0, sim_time, num_timesteps)

    for i in range(num_timesteps):
        plt.cla()
        ax.set_xlim(-1, 15)
        ax.set_ylim(-1, 15)
        
        for robot_state_history in robot_state_histories:
            circle = plt.Circle((robot_state_history[0, i], robot_state_history[1, i]), robot_radius, color='b')
            ax.add_artist(circle)

        plt.pause(0.01)

    plt.show()




def plot_multi_robot_and_obstacles(robot_histories, obstacles, robot_radius, num_steps, sim_time, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 21), ylim=(0, 21))
    ax.set_aspect('equal')
    ax.grid()

    lines = [ax.plot([], [], '--r')[0] for _ in range(robot_histories.shape[0])]
    robot_patches = [Circle((robot_histories[i, 0, 0], robot_histories[i, 1, 0]), robot_radius, facecolor='green', edgecolor='black') for i in range(robot_histories.shape[0])]
    
    obstacle_list = []
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle_patch = Circle((0, 0), robot_radius, facecolor='aqua', edgecolor='black')
        obstacle_list.append(obstacle_patch)

    def init():
        for robot_patch in robot_patches:
            ax.add_patch(robot_patch)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        for line in lines:
            line.set_data([], [])
        return robot_patches + lines + obstacle_list

    def animate(i):
        for j, robot_patch in enumerate(robot_patches):
            robot_patch.center = (robot_histories[j, 0, i], robot_histories[j, 1, i])
            lines[j].set_data(robot_histories[j, 0, :i], robot_histories[j, 1, :i])
        for k, obstacle_patch in enumerate(obstacle_list):
            obstacle_patch.center = (obstacles[0, i, k], obstacles[1, i, k])
        return robot_patches + lines + obstacle_list

    init()
    step = (sim_time / num_steps)
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # Save animation
    if not filename:
        return

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, num_steps), interval=200,
        blit=True, init_func=init)

    ani.save(filename, "ffmpeg", fps=30)


def plot_robot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color='aqua', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], '--r',)
    else:
        circle = plt.Circle((x, y), radius, color='green', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], 'blue')

    plt.gcf().gca().add_artist(circle)

def plot_robot_and_obstacles(robot, obstacles, robot_radius, num_steps, sim_time, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], '--r')

    robot_patch = Circle((robot[0, 0], robot[1, 0]),
                         robot_radius, facecolor='green', edgecolor='black')
    obstacle_list = []
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle = Circle((0, 0), robot_radius,
                          facecolor='aqua', edgecolor='black')
        obstacle_list.append(obstacle)

    def init():
        ax.add_patch(robot_patch)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        line.set_data([], [])
        return [robot_patch] + [line] + obstacle_list

    def animate(i):
        robot_patch.center = (robot[0, i], robot[1, i])
        for j in range(len(obstacle_list)):
            obstacle_list[j].center = (obstacles[0, i, j], obstacles[1, i, j])
        line.set_data(robot[0, :i], robot[1, :i])
        return [robot_patch] + [line] + obstacle_list

    init()
    step = (sim_time / num_steps)
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # Save animation
    if not filename:
        return

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, num_steps), interval=200,
        blit=True, init_func=init)

    ani.save(filename, "ffmpeg", fps=30)


def plot_robot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color='aqua', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], '--r',)
    else:
        circle = plt.Circle((x, y), radius, color='green', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], 'blue')

    plt.gcf().gca().add_artist(circle)