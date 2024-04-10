"""
Path Planning Sample Code with RRT and Dubins path

author: AtsushiSakai(@Atsushi_twi), Jigan Kim

modified by Jusuk Lee

"""
import random
import math
import copy
import numpy as np
import dubins_path_planning
import matplotlib.pyplot as plt
import matplotlib.patches as patches

show_animation = True

class Node():
    """
    RRT Node
    """
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.cost = 0.0
        self.parent = None

class RRT():
    def __init__(self, start, goal, obstacleList, randArea, goalSampleRate=5, maxIter=100):
        """
        start: Start Position [x, y, yaw]
        goal: Goal Position [x, y, yaw]
        obstacleList: obstacle Positions [[x, y, size], ...]
        randArea: Random Sampling Area (rectangular)
        """
        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])
        self.xrange = randArea['x']
        self.yrange = randArea['y']
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList
        
    def Planning(self, animation=True):
        self.nodeList = [self.start]
        for i in range(self.maxIter):
            done = False
            rnd, isgoal = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            
            newNode = self.steer(rnd, nind)
            
            if self.CollisionCheck(newNode, self.obstacleList):
                if isgoal is True:
                    done = True
                nearinds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearinds)
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)
                # add intermediate nodes
                if len(newNode.path_x) > 6:
                    newNode_ = copy.deepcopy(newNode)
                    # x,y,yaw 좌표의 중간 지점까지만 포함하도록 슬라이스.
                    idx = int(len(newNode.path_x) / 2)
                    newNode_.path_x = newNode_.path_x[0:idx+1]
                    newNode_.path_y = newNode_.path_y[0:idx+1]
                    newNode_.path_yaw = newNode_.path_yaw[0:idx+1]
                    newNode_.x = newNode_.path_x[-1]
                    newNode_.y = newNode_.path_y[-1]
                    newNode_.yaw = newNode_.path_yaw[-1]
                    deltax, deltay = [], []
                    # nodeList에 중간 노드를 추가
                    for j in range(len(newNode_.path_x)-1):
                        deltax.append(newNode_.path_x[j+1] - newNode_.path_x[j])
                        deltay.append(newNode_.path_y[j+1] - newNode_.path_y[j])
                    cost_ = np.sum(np.hypot(deltax, deltay))
                    newNode_.cost = cost_
                    self.nodeList.append(newNode_)
            if animation and i % 5 == 0:
                self.DrawGraph(rnd)

            print(f"iteration:{i}, nodes: {len(self.nodeList)}")
            
            if done:
                print("Goal!!")
                break
        lastIndex = self.get_best_last_index() # 목표에 가까우면서도 가장 목표의 방향에도 가까운 경로를 찾아 해당 경로의 인덱스를 반환

        if lastIndex is None:
            return None
        
        path = self.gen_final_course(lastIndex)
        return path
    
    def get_random_point(self):
        # Random sampling
        if random.randint(0, 100) > self.goalSampleRate:
            # self.nodeList에 있는 노드 중 하나를 무작위 선택하거나, 비용이 가장 높은 노드를 선택.
            # 비용이 높은 노드를 선택하면 덜 탐색된 영역으로 노드를 유도할 수 있다.
            if random.randint(0,1) == 0:
                costs = [node.cost for node in self.nodeList]
                maxind = costs.index(max(costs))
                ref_node = self.nodeList[maxind]
            else:
                ref_node = self.nodeList[random.randint(0, len(self.nodeList) - 1)]
            # 선택한 노드를 기준으로 x,y,yaw 범위를 설정. 
            minx = max(self.xrange[0], ref_node.x - 0.5*(self.xrange[1] - self.xrange[0]))
            maxx = min(self.xrange[1], ref_node.x + 0.5*(self.xrange[1] - self.xrange[0]))
            miny = max(self.yrange[0], ref_node.y - 0.5*(self.yrange[1] - self.yrange[0]))
            maxy = min(self.yrange[1], ref_node.y + 0.5*(self.yrange[1] - self.yrange[0]))
            minyaw = ref_node.yaw - math.pi/4
            maxyaw = ref_node.yaw + math.pi/4
            
            # 생성된 무작위 점이 장애물과 충돌하지 않는지 확인. 장애물이 없을 때까지 무작위 점을 생성하고 충돌 검사 반복
            isobstacle = True
            while isobstacle:
                rnd = [random.uniform(minx, maxx), random.uniform(miny, maxy), self.pi_2_pi(random.uniform(minyaw, maxyaw))]
                rnd_node = Node(rnd[0], rnd[1], rnd[2])
                rnd_node.path_x = [rnd[0]]
                rnd_node.path_y = [rnd[1]]
                rnd_node.path_yaw = [rnd[2]]
                isobstacle = not self.CollisionCheck(rnd_node, self.obstacleList)
            isgoal = False
        else:
            # Goal point sampling
            rnd = [self.end.x, self.end.y, self.end.yaw]
            isgoal = True
        
        node = Node(rnd[0], rnd[1], rnd[2])
        return node, isgoal
    
    def pi_2_pi(self, angle):
        # angle을 -pi ~ pi 범위로 변환. 361 deg = 1 deg이기 때문에.
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def CollisionCheck(self, node, obstacleList, dmin=0.5):
        for (ix, iy) in zip(node.path_x, node.path_y):
            d, _ = self.mindist_to_obstacles(ix, iy, obstacleList)
            # 점이 설정된 x 범위 또는 y 범위 밖에 있는지 여부 확인
            xcond = ix < self.xrange[0] or ix > self.xrange[1]
            ycond = iy < self.yrange[0] or iy > self.yrange[1]
            # 장애물과의 거리 계산
            if d < dmin or xcond or ycond:
                return False # collision
        return True
    
    def mindist_to_obstacles(self, x, y, obstacleList):
        """
        Find the minimum distance to the obstacles
        """
        minid = -1
        dmin = float("inf")
        for i, obstacle in enumerate(obstacleList):
            xcond = obstacle['x'] < x and x < obstacle['x'] + obstacle['xlen']
            ycond = obstacle['y'] < y and y < obstacle['y'] + obstacle['ylen']
            dcand = []
            if xcond and ycond: # inside the obstacle
                dmin = 0.0
                minid = i
                return dmin, minid
            elif xcond: # perpendicular to top/bottom sides
                dcand.append(min(abs(y-obstacle['y']), abs(y-obstacle['y'] - obstacle['ylen'])))
            elif ycond: # perpendicular to left/right sides
                dcand.append(min(abs(x-obstacle['x']), abs(x-obstacle['x'] + obstacle['xlen'])))
            
            # four coners
            dcand.append(np.hypot(x-obstacle['x'], y-obstacle['y']))
            dcand.append(np.hypot(x-obstacle['x'] - obstacle['xlen'], y-obstacle['y']))
            dcand.append(np.hypot(x-obstacle['x'], y-obstacle['y'] - obstacle['ylen']))
            dcand.append(np.hypot(x-obstacle['x'] - obstacle['xlen'], y-obstacle['y'] - obstacle['ylen']))
            if min(dcand) < dmin:
                dmin = min(dcand)
                minid = i
        return dmin, minid
    
    def GetNearestListIndex(self, nodeList, rnd):
        """
        node list에서 rnd 노드와 가장 가까운 노드의 인덱스 반환
        """
        # 각 노드에 대해 steer 함수를 사용해서 rnd 노드와 nodeList에 있는 노드 사이의 dubins path를 계산.
        # 그 중에서 가장 짧은 거리를 가지는 노드의 인덱스를 반환.
        clenlist = [self.steer(rnd, i).cost for i, _ in enumerate(nodeList)]
        minind = clenlist.index(min(clenlist))
        return minind
    
    def steer(self, rnd, nind):
        """
        Create a new node along the line connecting nearest node and random node with dubins path
        """
        curvature = 1.0
        
        nearestNode = self.nodeList[nind] # nearest node로 지정.
        
        # dubins path planning을 이용해서 무작위 노드 rnd와 nearest node 사이의 경로 생성.
        px, py, pyaw, mode, clen = dubins_path_planning.dubins_path_planning(nearestNode.x, nearestNode.y, nearestNode.yaw, rnd.x, rnd.y, rnd.yaw, curvature)
        
        # 새로운 노드 업데이트
        newNode = copy.deepcopy(nearestNode)
        newNode.x = px[-1] # 위에서 계산된 마지막 점을 새 노드로 선정.
        newNode.y = py[-1]
        newNode.yaw = pyaw[-1]
        
        newNode.path_x = px
        newNode.path_y = py
        newNode.path_yaw = pyaw
        newNode.cost += clen
        newNode.parent = nind
        
        return newNode
    
    def find_near_nodes(self, newNode):
        """
        RRT*에서 쓰인 것
        생성된 노드에 대하여 최적의 경로를 찾기 위해 주변 노드들을 재연결(rewire)하는 함수
        newNode에 대해 주변에 있는 노드들을 찾는 것이 목적.
        """
        nnode = len(self.nodeList)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode)) # r은 검색 반경을 설정.
        dlist = [
            (node.x - newNode.x) ** 2 + (node.y - newNode.y)**2 + (node.yaw - newNode.yaw)**2 for node in self.nodeList
        ]
        nearinds = [dlist.index(i) for i in dlist if i <= r**2]
        return nearinds
        
    def choose_parent(self, newNode, nearinds):
        """
        RRT*에서 쓰인 것
        newNode에 대해 최적의 부모 노드를 선택하는 과정이다. 충돌이 없고, 비용이 최소인 노드를 부모 노드로 설정
        """
        if not nearinds:
            return newNode
        dlist = []
        for i in nearinds:
            tNode = self.steer(newNode, i)
            if self.CollisionCheck(tNode, self.obstacleList):
                dlist.append(tNode.cost)
            else:
                dlist.append(float("inf"))
        
        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]
        
        if mincost == float("inf"):
            print("mincost is inf")
            return newNode
        
        newNode = self.steer(newNode, minind)
        return newNode
    
    def rewire(self, newNode, nearinds):
        nnode = len(self.nodeList)
        for i in nearinds:
            nearNode = self.nodeList[i]
            tNode = self.steer(nearNode, nnode-1)
            
            obstacleOK = self.CollisionCheck(tNode, self.obstacleList)
            improveCost = nearNode.cost > tNode.cost
            
            if obstacleOK and improveCost:
                self.nodeList[i] = tNode
    
    def DrawGraph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.path_x, node.path_y, "-g")
        
        for obstacle in pad_obstacles(self.obstacleList, padding=0.0):
            ax = plt.gca()
            rect = patches.Rectangle((obstacle['x'], obstacle['y']), obstacle['xlen'], obstacle['ylen'], linewidth=1, edgecolor='black', facecolor='r')
            ax.add_patch(rect)
        
        dubins_path_planning.plot_arrow(self.start.x, self.start.y, self.start.yaw)
        dubins_path_planning.plot_arrow(self.end.x, self.end.y, self.end.yaw)
        
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)
    
    def get_best_last_index(self):
        YAWTH = np.deg2rad(1.0)
        XYTH = 0.5 # TODO 다른점
        
        goalinds = []
        for (i, node) in enumerate(self.nodeList):
            if self.calc_dist_to_goal(node.x, node.y) <= XYTH:
                goalinds.append(i)
        
        # angle check
        fgoalinds = []
        for i in goalinds:
            if abs(self.nodeList[i].yaw - self.end.yaw) <= YAWTH:
                fgoalinds.append(i)
        
        if not fgoalinds:
            return None
        mincost = min([self.nodeList[i].cost for i in fgoalinds])
        for i in fgoalinds:
            if self.nodeList[i].cost == mincost:
                return i
        return None
    
    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])
    
    def gen_final_course(self, goalind):
        """최종 경로를 생성하는 역할"""
        path = [[self.end.x, self.end.y]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            goalind = node.parent
        path.append([self.start.x, self.start.y])
        return path
        
        
        
            
            
            
            
            
def pad_obstacles(obstacles, padding=0.0):
    padded_obstacles = []
    for obs in obstacles:
        padded_obstacle = obs.copy()
        padded_obstacle['x'] -= padding
        padded_obstacle['y'] -= padding
        padded_obstacle['xlen'] += 2*padding
        padded_obstacle['ylen'] += 2*padding
        padded_obstacles.append(padded_obstacle)
    return padded_obstacles


def main():
    print("Start rrt star with dubins planning")
    obstacleList = [
        {'x': 0.0, 'y': 4.0, 'xlen': 1.0, 'ylen': 7.0},
        {'x': 4.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 4.0},
        {'x': 4.0, 'y': 7.0, 'xlen': 3.0, 'ylen': 3.0},
        {'x': 4.0, 'y': 14.0, 'xlen': 13.0, 'ylen': 1.0},
        {'x': 5.0, 'y': 10.0, 'xlen': 1.0, 'ylen': 4.0},
        {'x': 8.0, 'y': 0.0, 'xlen': 13.0, 'ylen': 1.0},
        {'x': 8.0, 'y': 3.0, 'xlen': 2.0, 'ylen': 2.0},
        {'x': 9.0, 'y': 12.0, 'xlen': 1.0, 'ylen': 2.0},
        {'x': 12.0, 'y': 1.0, 'xlen': 1.0, 'ylen': 11.0},
        {'x': 16.0, 'y': 4.0, 'xlen': 2.0, 'ylen': 8.0},
        {'x': 18.0, 'y': 7.0, 'xlen': 7.0, 'ylen': 1.0},
        {'x': 20.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 4.0},
        {'x': 22.0, 'y': 14.0, 'xlen': 3.0, 'ylen': 1.0}, 
        {'x': 24.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 14.0}
    ] # rectangular obstacle
    
    padded_obstacleList = pad_obstacles(obstacleList, padding=np.sqrt(2)/2.0)
    
    start = [2.0, 15.0, np.deg2rad(0)]
    end = [23.0, 1.0, np.deg2rad(0)]
    
    rrt = RRT(start, end, randArea={'x': [0.0, 25.0], 'y': [0.0, 15.0]}, obstacleList=obstacleList, maxIter=5000)
    path = rrt.Planning(animation=show_animation)
    
    # Draw final path
    if show_animation:
        rrt.DrawGraph()
        plt.plot([x for (x, y) in path], [y for (x,y) in path], '-r')
        plt.grid(True)
        plt.pause(0.01)
        
        plt.show()
if __name__ == "__main__":
    main()