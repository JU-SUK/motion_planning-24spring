from P2.dubins_path_planning import dubins_path_planning_from_origin as dubins_path_planning


"""
Path plan algorithm, include: A*, APF(Artificial Potential Field

author: ShuiXinYun

modified by Jusuk Lee

Ref:
https://github.com/ShuiXinYun/Path_Plan/tree/master
"""
class Array2D:
    def __init__ (self, w, h, mapdata=[]):
        self.w = w
        self.h = h
        if mapdata:
            self.data = mapdata
        else:
            self. data = [[0 for y in range(h)] for x in range(w)]
            
    def showArray2D(self):
        for y in range(self.h):
            for x in range(self.w):
                print(self.data[x][y], end='')
            print("")
    
    def __getitem__(self, item):
        return self.data[item]
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False
    
    def __str__(self):
        return f'(x: {self.x}, y: {self.y})'

class AStar:
    class Node:
        def __init__(self, point, endPoint, g=0):
            self.point = point
            self.father = None
            self.g = g
            self.h = (abs(endPoint.x - point.x) + abs(endPoint.y - point.y)) * 10
    
    def __init__(self, map2d, startPoint, endPoint, passTag=0):
        self.openList = []
        self.closeList = []
        self.map2d = map2d
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.passTag = passTag
    
    def getMinNode(self):
        """
        return: Node
        """
        currentNode = self.openList[0]
        for node in self.openList:
            if node.g + node.h < currentNode.g + currentNode.h:
                currentNode = node
        return currentNode
    
    def pointInCloseList(self, point):
        for node in self.closeList:
            if node.point == point:
                return True
        return False
    
    def pointInOpenList(self, point):
        for node in self.openList:
            if node.point == point:
                return node
        return None
    
    def endPointInCloseList(self):
        for node in self.openList:
            if node.point == self.endPoint:
                return node
        return None
    
    def searchNear(self, minF, offsetX, offsetY):
        import math
        # if minF.point.x + offsetX < 0 or minF.point.x + offsetX > self.map2d.w -1 or minF.point.y + offsetY < 0 or minF.point.y + offsetY > self.map2d.h -1:
        #     return
        # if self.map2d[minF.point.x + offsetX][minF.point.y + offsetY] != self.passTag:
        #     return
        # if self.pointInCloseList(Point(minF.point.x + offsetX, minF.point.y + offsetY)):
        #     return
        
        # if offsetX == 0 or offsetY == 0:\
        #     step = 10
        # else:
        #     step = 14
        
        currentNode = self.pointInOpenList(Point(minF.point.x + offsetX, minF.point.y + offsetY))
        if currentNode is None:
            newNode = AStar.Node(Point(minF.point.x + offsetX, minF.point.y + offsetY), self.endPoint)
        else:
            newX = currentNode.point.x
        potential_orientations = [0, math.pi/4, math.pi/2, -math.pi/4, -math.pi/2]
        c = 1.0
        for orientation in potential_orientations:
            for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
                nexX = currentNode.point.x + dx
                newY = currentNode.point.y + dy
                newYaw = orientation
                
                px, py, pyaw, mode, clen = dubins_path_planning(currentNode.point.x, currentNode.point.y, currentNode.yaw, nexX, newY, newYaw, c)
                
                isFeasible = True
                
                if isFeasible:
                    newCost = currentNode.g + clen
                    existingNode = self.pointInOpenList(Point(px, py))
                    if existingNode and newCost < existingNode.g:
                        existingNode.g = newCost
                        existingNode.father = currentNode
                    elif not existingNode:
                        newNode = AStar.Node(Point(px, py), self.endPoint, g=newCost)
                        newNode.father = currentNode
                        self.openList.append(newNode)
            
        
        # currentNode = self.pointInOpenList(Point(minF.point.x + offsetX, minF.point.y + offsetY))
        # if not currentNode:
        #     currentNode = AStar.Node(Point(minF.point.x + offsetX, minF.point.y + offsetY), self.endPoint, g=minF.g + step)
        #     currentNode.father = minF
        #     self.openList.append(currentNode)
        #     return
        
        # if minF.g + step < currentNode.g:
        #     currentNode.g = minF.g + step
        #     currentNode.father = minF
    
    def setNearOnce(self, x, y):
        offset = 1
        points = [[-offset, offset], [0, offset], [offset, offset], [-offset, 0], [offset, 0], [-offset, -offset], [0, -offset], [offset, -offset]]
        for point in points:
            if 0 <= x + point[0] < self.map2d.w and 0 <= y + point[1] < self.map2d.h:
                self.map2d.data[x + point[0]][y + point[1]] = 1
    
    def expansion(self, offset=0):
        for i in range(offset):
            barrierxy = list()
            for x in range(self.map2d.w):
                for y in range(self.map2d.h):
                    if self.map2d.data[x][y] not in [self.passTag, 'S', 'E']:
                        barrierxy.append([x, y])
            for xy in barrierxy:
                self.setNearOnce(xy[0], xy[1])
                
    def start(self):
        # Step 1: Put the starting point into the open list
        startNode = AStar.Node(self.startPoint, self.endPoint)
        self.openList.append(startNode)
        
        # Step 2: Start the main loop
        while True:
            # Step 3: Find the point with the smallest F value in the open list
            minF = self.getMinNode()
            self.closeList.append(minF)
            self.openList.remove(minF)
            
            # Step 4: For the point in the vicinity of the current point
            self.searchNear(minF,  -1, 1)
            self.searchNear(minF,  0, 1)
            self.searchNear(minF,  1, 1)
            self.searchNear(minF,  -1, 0)
            self.searchNear(minF,  1, 0)
            self.searchNear(minF,  -1, -1)
            self.searchNear(minF,  0, -1)
            self.searchNear(minF,  1, -1)
            
            # Step 5: Determine whether the end point is in the open list
            point = self.endPointInCloseList()
            if point:
                cPoint = point
                pathList = []
                while True:
                    if cPoint.father:
                        pathList.append(cPoint.point)
                        cPoint = cPoint.father
                    else:
                        return list(reversed(pathList))
            # Step 6: Determine whether the open list is empty
            if len(self.openList) == 0:
                return None

if __name__ == '__main__':
    map2d = Array2D(15, 15)
    
    for i in range(6):
        map2d[3][i] = 1
    for i in range(4):
        map2d[10][i] = 1
    print("Input map:")
    map2d.showArray2D()
    
    pStart, pEnd = Point(0, 0), Point(14, 0)
    aStar = AStar(map2d, pStart, pEnd)
    aStar.expansion(offset=0)
    aStar.map2d.showArray2D()
    
    pathList = aStar.start()
    if pathList:
        print("---Route Node:")
        for point in pathList:
            map2d[point.x][point.y] = "#"
            print(f'{pathList.index(point)}: {point}', end=' ')
        print("---Route:")
        map2d[pStart.x][pStart.y], map2d[pEnd.x][pEnd.y] = "S", "E"
        map2d.showArray2D()
    else:
        print("No path found!")    
        