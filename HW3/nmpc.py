"""
Collision avoidance using Nonlinear Model-Predictive Control with multiple agents

author: Ashwin Bose (atb033@github.com)
"""

from utils.multi_robot_plot import plot_robots_and_obstacles
import numpy as np
from scipy.optimize import minimize, Bounds
import time

SIM_TIME = 10.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2
NUM_AGENTS = 4

# collision cost parameters
Qc = 5.
kappa = 4.

# nmpc parameters
HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3
upper_bound = [(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2


def simulate(filename):
    starts = np.array([[5, 5], [10, 5], [5, 10], [10, 10]])
    p_desired = np.array([[10, 10], [5, 10], [10, 5], [5, 5]])

    robots_state = starts
    robots_state_history = np.empty((NUM_AGENTS, 4, NUMBER_OF_TIMESTEPS))

    for i in range(NUM_AGENTS):
        robots_state_history[i, :2, 0] = robots_state[i]

    for t in range(NUMBER_OF_TIMESTEPS):
        for i in range(NUM_AGENTS):
            other_agents = np.delete(robots_state, i, axis=0)
            xref = compute_xref(robots_state[i], p_desired[i],
                                HORIZON_LENGTH, NMPC_TIMESTEP)
            vel, velocity_profile = compute_velocity(
                robots_state[i], other_agents, xref)
            robots_state[i] = update_state(robots_state[i], vel, TIMESTEP)
            robots_state_history[i, :2, t] = robots_state[i]

    plot_robots_and_obstacles(
        robots_state_history, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


def compute_velocity(robot_state, other_agents, xref):
    """
    Computes control velocity of the copter
    """
    u0 = np.random.rand(2*HORIZON_LENGTH)
    def cost_fn(u): return total_cost(
        u, robot_state, other_agents, xref)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start)
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape((2*number_of_steps))


def total_cost(u, robot_state, other_agents, xref):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    c2 = total_collision_cost(x_robot, other_agents)
    total = c1 + c2
    return total


def tracking_cost(x, xref):
    return np.linalg.norm(x-xref)


def total_collision_cost(robot, other_agents):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        rob = robot[2 * i: 2 * i + 2]
        for j in range(len(other_agents)):
            other_agent = other_agents[j]
            other_position = other_agent[:2]
            total_cost += collision_cost(rob, other_position)
    return total_cost


def collision_cost(x0, x1):
    """
    Cost of collision between two robot_state
    """
    d = np.linalg.norm(x0 - x1)
    cost = Qc / (1 + np.exp(kappa * (d - 2*ROBOT_RADIUS)))
    return cost


def update_state(x0, u, timestep):
    """
    Computes the states of the system after applying a sequence of control signals u on
    initial state x0
    """
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))

    new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * timestep

    return new_state

if __name__ == "__main__":
    simulate("nmpc.mp4")
