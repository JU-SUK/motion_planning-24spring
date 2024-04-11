"""
Path plan algorithm, include: A*, APF(Artificial Potential Field

author: ShuiXinYun

modified by Jusuk Lee

Ref:
https://github.com/ShuiXinYun/Path_Plan/tree/master
"""
import P3 as P3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def visualize_map(map2d, start, end, path):
    cmap = mcolors.ListedColormap(['white', 'black', 'red', 'green', 'blue'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    
    grid = np.array([[0 if cell ==0 else 1 for cell in row] for row in map2d.data])
    grid[start.x][start.y] = 3
    grid[end.x][end.y] = 4
    
    for point in path:
        grid[point.x][point.y] = 2
        
    for point in path[:-1]:
        grid[point.x][point.y] = 4
        
    grid = grid.T
    
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks([x-0.5 for x in range(1 + max(len(row) for row in grid))])
    ax.set_yticks([y-0.5 for y in range(1 + len(grid))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def readmap():
    mapinfo = dict()
    l_t = list()
    w, h = 0, 0
    with open('P3_map.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            l = [int(s) if s != 'S' and s != 'E' else s for s in line]
            l_t.append(l)
            h += 1
            w = len(l)

            for c in l:
                if c == 'S':
                    mapinfo.update({'start': P3.Point(l.index(c), h - 1)})
                    l_t[-1][l.index(c)] = 0

                if c == 'E':
                    mapinfo.update({'end': P3.Point(l.index(c), h - 1)})
                    l_t[-1][l.index(c)] = 0
    l_t_t = list()
    for i in range(w):
        t = list()
        for j in range(h):
            t.append(l_t[j][i])
        l_t_t.append(t)
    map2d = P3.Array2D(w, h, l_t_t)
    mapinfo.update({'map2d': map2d})
    return mapinfo

if __name__ == "__main__":
    mapinfo = readmap()
    aStar = P3.AStar(mapinfo['map2d'], mapinfo['start'], mapinfo['end'])
    print(f'start:{mapinfo["start"]}, end:{mapinfo["end"]}')
    print('------------------\nMap:')
    
    mapinfo['map2d'][mapinfo['start'].x][mapinfo['start'].y] = 'S'
    mapinfo['map2d'][mapinfo['end'].x][mapinfo['end'].y] = 'E'
    mapinfo['map2d'].showArray2D()
    print('------------------\n map after expansion:')
    aStar.expansion(offset=0)
    aStar.map2d.showArray2D()
    
    # Start the search
    mapinfo['map2d'][mapinfo['start'].x][mapinfo['start'].y] = 0
    mapinfo['map2d'][mapinfo['end'].x][mapinfo['end'].y] = 0
    pathList = aStar.start()
    
    if pathList:
        print("----------- \nRoute Node:")
        for point in pathList:
            mapinfo['map2d'][point.x][point.y] = '#'
            print(f'{pathList.index(point)}: {point}', end=' ')
        print("----------- \nRoute:")
        mapinfo['map2d'][mapinfo['start'].x][mapinfo['start'].y] = 'S'
        mapinfo['map2d'][mapinfo['end'].x][mapinfo['end'].y] = 'E'
        
        mapinfo['map2d'].showArray2D()
    else:
        print("No path found!")
        
    visualize_map(mapinfo['map2d'], mapinfo['start'], mapinfo['end'], pathList)
        