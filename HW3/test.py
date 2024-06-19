import numpy as np
import matplotlib.pyplot as plt

class DubinsCar:
    def __init__(self, x0, y0, theta0, v, L):
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.v = v
        self.L = L

    def update(self, u, dt):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.L) * u * dt

    def state(self):
        return self.x, self.y, self.theta

def compute_control(state, goal):
    x, y, theta = state
    goal_x, goal_y = goal

    alpha = np.arctan2(goal_y - y, goal_x - x)
    u = alpha - theta
    u = (u + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    return u

def simulate_dubins_car_schedule(schedules, v, L, dt):
    agents = []
    goals = []
    for schedule in schedules:
        x0, y0 = schedule[0]['x'], schedule[0]['y']
        theta0 = np.arctan2(schedule[1]['y'] - y0, schedule[1]['x'] - x0) if len(schedule) > 1 else 0
        agents.append(DubinsCar(x0, y0, theta0, v, L))
        goals.append(schedule)

    histories = [[] for _ in range(len(schedules))]
    for t in range(len(schedules[0])):
        for i, agent in enumerate(agents):
            if t < len(schedules[i]):
                goal = (schedules[i][t]['x'], schedules[i][t]['y'])
                while np.linalg.norm([agent.x - goal[0], agent.y - goal[1]]) > 0.1:
                    u = compute_control(agent.state(), goal)
                    agent.update(u, dt)
                    histories[i].append(agent.state())
            else:
                histories[i].append(agent.state())

    return histories

def plot_dubins_car_trajectories(histories, schedules):
    plt.figure()
    for i, history in enumerate(histories):
        history = np.array(history)
        plt.plot(history[:, 0], history[:, 1], label=f'Agent {i+1}')
        schedule = schedules[i]
        plt.scatter([point['x'] for point in schedule], [point['y'] for point in schedule])
        plt.scatter(schedule[0]['x'], schedule[0]['y'], marker='o', color='green')  # Start point
        plt.scatter(schedule[-1]['x'], schedule[-1]['y'], marker='x', color='red')  # End point

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Dubins Car Trajectories')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    schedule = {
        'agent1': [
            {'t': 0, 'x': 0, 'y': 0},
            {'t': 1, 'x': 1, 'y': 0},
            {'t': 2, 'x': 2, 'y': 0},
            {'t': 3, 'x': 3, 'y': 0},
            {'t': 4, 'x': 3, 'y': 1},
            {'t': 5, 'x': 3, 'y': 2},
            {'t': 6, 'x': 3, 'y': 3},
            {'t': 7, 'x': 3, 'y': 4},
            {'t': 8, 'x': 3, 'y': 5},
            {'t': 9, 'x': 3, 'y': 6},
            {'t': 10, 'x': 4, 'y': 6},
            {'t': 11, 'x': 5, 'y': 6},
            {'t': 12, 'x': 6, 'y': 6},
            {'t': 13, 'x': 7, 'y': 6},
            {'t': 14, 'x': 7, 'y': 7},
            {'t': 15, 'x': 8, 'y': 7},
            {'t': 16, 'x': 8, 'y': 8},
            {'t': 17, 'x': 8, 'y': 9},
            {'t': 18, 'x': 9, 'y': 9}
        ],
        'agent2': [
            {'t': 0, 'x': 9, 'y': 0},
            {'t': 1, 'x': 9, 'y': 1},
            {'t': 2, 'x': 9, 'y': 2},
            {'t': 3, 'x': 9, 'y': 3},
            {'t': 4, 'x': 8, 'y': 3},
            {'t': 5, 'x': 8, 'y': 4},
            {'t': 6, 'x': 7, 'y': 4},
            {'t': 7, 'x': 7, 'y': 5},
            {'t': 8, 'x': 6, 'y': 5},
            {'t': 9, 'x': 5, 'y': 5},
            {'t': 10, 'x': 4, 'y': 5},
            {'t': 11, 'x': 3, 'y': 5},
            {'t': 12, 'x': 2, 'y': 5},
            {'t': 13, 'x': 1, 'y': 5},
            {'t': 14, 'x': 1, 'y': 6},
            {'t': 15, 'x': 1, 'y': 7},
            {'t': 16, 'x': 1, 'y': 8},
            {'t': 17, 'x': 0, 'y': 8},
            {'t': 18, 'x': 0, 'y': 9}
        ],
        'agent3': [
            {'t': 0, 'x': 0, 'y': 9},
            {'t': 1, 'x': 1, 'y': 9},
            {'t': 2, 'x': 1, 'y': 8},
            {'t': 3, 'x': 1, 'y': 7},
            {'t': 4, 'x': 1, 'y': 6},
            {'t': 5, 'x': 1, 'y': 5},
            {'t': 6, 'x': 2, 'y': 5},
            {'t': 7, 'x': 2, 'y': 4},
            {'t': 8, 'x': 2, 'y': 3},
            {'t': 9, 'x': 3, 'y': 3},
            {'t': 10, 'x': 4, 'y': 3},
            {'t': 11, 'x': 5, 'y': 3},
            {'t': 12, 'x': 6, 'y': 3},
            {'t': 13, 'x': 6, 'y': 2},
            {'t': 14, 'x': 6, 'y': 1},
            {'t': 15, 'x': 7, 'y': 1},
            {'t': 16, 'x': 7, 'y': 0},
            {'t': 17, 'x': 8, 'y': 0},
            {'t': 18, 'x': 9, 'y': 0}
        ],
        'agent4': [
            {'t': 0, 'x': 9, 'y': 9},
            {'t': 1, 'x': 8, 'y': 9},
            {'t': 2, 'x': 7, 'y': 9},
            {'t': 3, 'x': 6, 'y': 9},
            {'t': 4, 'x': 5, 'y': 9},
            {'t': 5, 'x': 5, 'y': 8},
            {'t': 6, 'x': 4, 'y': 8},
            {'t': 7, 'x': 3, 'y': 8},
            {'t': 8, 'x': 3, 'y': 7},
            {'t': 9, 'x': 2, 'y': 7},
            {'t': 10, 'x': 1, 'y': 7},
            {'t': 11, 'x': 1, 'y': 6},
            {'t': 12, 'x': 1, 'y': 5},
            {'t': 13, 'x': 1, 'y': 4},
            {'t': 14, 'x': 0, 'y': 4},
            {'t': 15, 'x': 0, 'y': 3},
            {'t': 16, 'x': 0, 'y': 2},
            {'t': 17, 'x': 0, 'y': 1},
            {'t': 18, 'x': 0, 'y': 0}
        ]
    }

    schedules = [schedule['agent1'], schedule['agent2'], schedule['agent3'], schedule['agent4']]
    v = 1.0
    L = 2.0
    dt = 0.1

    histories = simulate_dubins_car_schedule(schedules, v, L, dt)
    plot_dubins_car_trajectories(histories, schedules)
