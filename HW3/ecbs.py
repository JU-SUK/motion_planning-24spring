import sys
sys.path.insert(0, '../')
import argparse
import yaml
from math import fabs
import numpy as np
from itertools import combinations
from copy import deepcopy

from a_star import AStar
import math

class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __str__(self):
        return str((self.x, self.y))

class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location.x) + str(self.location.y))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class Environment_planning(object):
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self)

    def get_neighbors(self, state):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        return neighbors


    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y) not in self.obstacles

    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        eps = 3.0 # if this value were 1.0, smae with cbs
        # explore_weight = 1.0 # probability of having an enemy in each cell
        goal = self.agent_dict[agent_name]["goal"]
        return eps * (fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y))
        # return eps * (fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)) - explore_weight * np.sum(state.location.probability_each_cell)
        # return eps * (np.sqrt((state.location.x - goal.location.x) ** 2 + (state.location.y - goal.location.y) ** 2))

    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = State(0, Location(agent['goal'][0], agent['goal'][1]))

            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def compute_solution(self):
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution = self.a_star.search(agent)
            if not local_solution:
                return False
            solution.update({agent:local_solution})
        return solution

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost

class ECBS(object):
    def __init__(self, environment):
        self.env = environment
        self.open_set = set()
        self.closed_set = set()
    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)

        self.open_set |= {start}

        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                print("solution found")

                return self.generate_plan(P.solution)

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # TODO: ending condition
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return {}

    def generate_plan(self, solution):
        plan = {}
        plan_x = {}
        plan_y = {}
        for agent, path in solution.items():
            # # path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path] # For YAML File
            # # path_dict_list_x = [{'x':state.location.x} for state in path]
            # path_dict_list_x = [state.location.x for state in path] # For integrating with RL, CTRL
            # # path_dict_list_y = [{'y':state.location.y} for state in path]
            # path_dict_list_y = [state.location.y for state in path] # For integrating with RL, CTRL
            # # plan[agent] = path_dict_list # For YAML File
            # plan_x[agent] = path_dict_list_x
            # plan_y[agent] = path_dict_list_y
            # plan[agent] = [plan_x[agent], plan_y[agent]]
            
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan
    
    
    
class DubinsPath:
    def __init__(self, turning_radius):
        self.turning_radius = turning_radius
    
    def _mod2pi(self, theta):
        return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)    
    def _calculate_dubins_parameters(self, start, goal):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        D = math.sqrt(dx**2 + dy**2)
        d = D / self.turning_radius

        theta = self._mod2pi(math.atan2(dy, dx))
        alpha = self._mod2pi(start[2] - theta)
        beta = self._mod2pi(goal[2] - theta)

        return alpha, beta, d

    def _LSL(self, alpha, beta, d):
        tmp0 = d + math.sin(alpha) - math.sin(beta)
        p_squared = 2 + (d * d) - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) - math.sin(beta)))
        if p_squared < 0:
            return None, float('inf')
        tmp1 = math.atan2((math.cos(beta) - math.cos(alpha)), tmp0)
        t = self._mod2pi(-alpha + tmp1)
        p = math.sqrt(p_squared)
        q = self._mod2pi(beta - tmp1)
        return [t, p, q], t + p + q

    def _RSR(self, alpha, beta, d):
        tmp0 = d - math.sin(alpha) + math.sin(beta)
        p_squared = 2 + (d * d) - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(beta) - math.sin(alpha)))
        if p_squared < 0:
            return None, float('inf')
        tmp1 = math.atan2((math.cos(alpha) - math.cos(beta)), tmp0)
        t = self._mod2pi(alpha - tmp1)
        p = math.sqrt(p_squared)
        q = self._mod2pi(-beta + tmp1)
        return [t, p, q], t + p + q

    def calculate_shortest_path(self, start, goal):
        alpha, beta, d = self._calculate_dubins_parameters(start, goal)

        paths = [self._LSL(alpha, beta, d), self._RSR(alpha, beta, d)]
        min_length = float('inf')
        best_path = None

        for path, length in paths:
            if length < min_length:
                min_length = length
                best_path = path

        return best_path, min_length

    def generate_dubins_path(self, start, goal):
        path, length = self.calculate_shortest_path(start, goal)
        if path is None:
            return []

        points = []
        step_size = 1.0

        t, p, q = path
        for i in np.arange(0, t, step_size):
            x = start[0] + self.turning_radius * math.cos(start[2] + i)
            y = start[1] + self.turning_radius * math.sin(start[2] + i)
            points.append((x, y))

        for i in np.arange(0, p, step_size):
            x = points[-1][0] + step_size * math.cos(start[2] + t)
            y = points[-1][1] + step_size * math.sin(start[2] + t)
            points.append((x, y))

        for i in np.arange(0, q, step_size):
            x = points[-1][0] + self.turning_radius * math.cos(start[2] + t + i)
            y = points[-1][1] + self.turning_radius * math.sin(start[2] + t + i)
            points.append((x, y))

        return points

class EnvironmentPlanningDubins(Environment_planning):
    def __init__(self, dimension, agents, obstacles, turning_radius):
        super().__init__(dimension, agents, obstacles)
        self.dubins_path = DubinsPath(turning_radius)
        self.turning_radius = turning_radius
    def get_dubins_path(self, start, goal):
        return self.dubins_path.generate_dubins_path(start, goal)
    
    def compute_dubins_solution(self, solution):
        dubins_solution = {}
        for agent, path in solution.items():
            dubins_path = []
            for i in range(len(path) - 1):
                start = path[i].location
                goal = path[i + 1].location
                dubins_path_segement = self.get_dubins_path(start, goal)
                dubins_path.extend(dubins_path_segement)
            dubins_solution[agent] = dubins_path
        return dubins_solution
    
    def search(self):
        ecbs_solution = super().search()
        if not ecbs_solution:
            return {}
        return self.compute_dubins_solution(ecbs_solution)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    args = parser.parse_args()

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)


    ##### For integrating with RL, CTRL #####
    # dimension = np.zeros((2,))
    # dimension[0] = map.grid_size * 10
    # dimension[1] = map.grid_size * 10

    # agents = {"name":['agent0', 'agent1', 'agent2', 'agent3'], "start":[], "goal":[]}
    # agents['start'].add(agent_positions)
    # agents['goal'].add(actions.subgoals)

    # obstacles = map.obstacle_area
    #########################################


    ##### For YAML File #####
    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']
    turning_radius = param.get('turning_radius', 1.0)
    #########################

    env_planning_dubins = EnvironmentPlanningDubins(dimension, agents, obstacles, turning_radius)

    # Searching
    ecbs = ECBS(env_planning_dubins)
    solution = ecbs.search()


    ##### For integrating with RL, CTRL #####
    # tank1_path_xs = solution['agent0'][0]
    # tank1_path_ys = solution['agent0'][1]
    # tank2_path_xs = solution['agent1'][0]
    # tank2_path_ys = solution['agent1'][1]
    # tank3_path_xs = solution['agent2'][0]
    # tank3_path_ys = solution['agent2'][1]
    # tank4_path_xs = solution['agent3'][0]
    # tank4_path_ys = solution['agent3'][1]
    #########################################


    if not solution:
        print(" Solution not found" )
        return
    
    plan = ecbs.generate_plan(solution)

    # Write to output file
    output = dict()
    output["schedule"] = plan
    output["cost"] = env_planning_dubins.compute_solution_cost(solution)
    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml, default_flow_style=False)




if __name__ == "__main__":
    main()