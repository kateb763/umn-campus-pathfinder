import collections
import heapq
import math
import json
import random

def euclidean_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

class MapGraph:
    """A Graph"""

    def __init__(self, json_file):
        """Constructor takes in a file path and loads in Vertices and Edges"""
        self.graph = None
        with open('graph.json') as json_file:
            self.graph = json.load(json_file)

    def get_vertices(self):
        """Returns vertices"""
        return list(self.graph["E"].keys())

    def get_neighbors(self, v):
        """Returns edges"""
        return self.graph["E"][v]

    def get_position(self, v):
        """Returns position of vertex"""
        return self.graph["V"][v]["position"]
    
    def find_closest_vertex(self, point):
        """Returns the closest vertex to point"""
        verts = self.get_vertices()
        closest = verts[0]
        min_dist = math.inf
        for v in verts:
            dist = euclidean_dist(point, self.get_position(v))
            if (min_dist > dist):
                min_dist = dist
                closest = v
        return closest


def point_to_point(start, dest):
    return [start, dest]

def fly(start, dest):
    path = []
    path.append(start)
    dist = euclidean_dist(start, dest)
    ds = 1.0/20.0
    d = ds
    while d < 1.0:
        d = d + ds
        path.append([(dest[0]-start[0])*d+start[0], start[1] + 50*((-(2*d-1)**2+1)), (dest[2]-start[2])*d+start[2]])
    path.append(dest)
    return path

def random_graph(start, dest):
    graph = MapGraph('graph.json')
    
    start_node = graph.find_closest_vertex(start)
    node = random.choice(graph.get_vertices())
    path = fly(graph.get_position(start_node), graph.get_position(node))
    visitied = {}

    for i in range(0, 1000):
        neighbors = graph.get_neighbors(node)
        for neighbor in neighbors:
            if (neighbor not in visitied):
                node = neighbor
                path.append(graph.get_position(node))
                visitied[node] = True
                break
    return path

def breadth_first(start, dest):
    graph = MapGraph('graph.json')

    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)

    visited = set()
    visited.add(start_node)
    parent = {start_node: None}
    queue = collections.deque([start_node])

    while queue:
        curr_node = queue.popleft()
        if curr_node == dest_node:
            break

        for neighbor in graph.get_neighbors(curr_node):
            if(neighbor not in visited):
                visited.add(neighbor)
                parent[neighbor] = curr_node
                queue.append(neighbor)
    
    curr = dest_node
    path = []
    while curr is not None:
        path.append(graph.get_position(curr))
        curr = parent[curr]
    #this way it's start -> destination
    path.reverse()

    for i in range(1, len(path)):
        fly(path[i-1], path[i])

    return path

def breadth_first_hub(start, dest):
    graph = MapGraph('graph.json')
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    start_pos = graph.get_position(start_node)
    dest_pos = graph.get_position(dest_node)

    #identify all hubs
    hubs = []
    for node in graph.get_vertices():
        if len(graph.get_neighbors(node)) >= 5:
            pos = graph.get_position(node)
            dist_to_start = euclidean_dist(pos, start_pos)
            dist_to_dest = euclidean_dist(pos, dest_pos)
            max_dist = max(dist_to_start, dist_to_dest)#ensure hub is close to both points
            hubs.append((node, max_dist, dist_to_dest, pos))

    #1st sort hubs by closeness to both points then by distance to destination
    hubs.sort(key=lambda x: (x[1], x[2]))
    top_hubs = hubs[:3]
    #then sort hubs by distance to destination(furthest first)
    top_hubs.sort(key=lambda x: x[2], reverse=True)
    waypoints = [start_pos] + [hub[3] for hub in top_hubs] + [dest_pos]#construct path w/ the hubs
    path = []
    
    #connect each segment using BFS
    for i in range(len(waypoints)-1):
        segment_path = breadth_first(waypoints[i], waypoints[i+1])
        if path:
            path.extend(segment_path[1:])#avoid duplicate points by skipping first point of segment

        else:
            path.extend(segment_path)
    
    for i in range(1, len(path)):
        fly(path[i-1], path[i])
    
    return path


def depth_first(start, dest):
    path = []
    graph = MapGraph('graph.json')
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    prev_node = start_node
    visited = set()
    visited.add(start_node)
    stack = [start_node]  

    while stack:
        curr_node = stack.pop()
        if prev_node is not None:
            fly(graph.get_position(prev_node), graph.get_position(curr_node))

        path.append(graph.get_position(curr_node))
        if curr_node == dest_node:
            break
        #here reversed because DFS
        for neighbor in reversed(graph.get_neighbors(curr_node)):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
        prev_node = curr_node
    return path

def depth_first_best(start, dest):
    path = []
    graph = MapGraph('graph.json')
    prev_node = None
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    visited = set()
    visited.add(start_node)
    stack = [start_node]
    dest_pos = graph.get_position(dest_node)

    while stack:
        curr_node = stack.pop()
        if prev_node is not None:
            fly(graph.get_position(prev_node), graph.get_position(curr_node))

        path.append(graph.get_position(curr_node))
        if curr_node == dest_node:
            break

        neighbors = graph.get_neighbors(curr_node)
        # sort neighbors by Euclidean distance to destination
        neighbors.sort(
            key=lambda node: euclidean_dist(graph.get_position(node), dest_pos),
            reverse=True  
        )

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

        prev_node = curr_node

    return path

def bellman_ford(start, dest):
    graph = MapGraph('graph.json')
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    vertices = graph.get_vertices()
    dist = {node: float('inf') for node in vertices}
    prev = {node: None for node in vertices}
    dist[start_node] = 0

    #relaxation step..
    for _ in range(len(vertices) - 1):
        updated = False
        for vertex in vertices:
            if dist[vertex] == float('inf'):
                continue
            for neighbor in graph.get_neighbors(vertex):
                weight = euclidean_dist(graph.get_position(vertex), graph.get_position(neighbor))
                if dist[neighbor] > dist[vertex] + weight:
                    dist[neighbor] = dist[vertex] + weight
                    prev[neighbor] = vertex
                    updated = True
        #early exit if there are no updates
        if not updated:
            break  
        
    path = []
    current_node = dest_node
    
    while current_node is not None:
        path.append(graph.get_position(current_node))
        current_node = prev[current_node]

    path.reverse()
    
    for i in range(1, len(path)):
        fly(path[i-1], path[i])

    return path

def bellman_ford_negative(start, dest):
    # graph = MapGraph('graph.json')
    # start_node = graph.find_closest_vertex(start)
    # dest_node = graph.find_closest_vertex(dest)
    # vertices = graph.get_vertices()
    # dist = {node: float('inf') for node in vertices}
    # prev = {node: None for node in vertices}
    # dist[start_node] = 0

    # #relaxation step..
    # for _ in range(len(vertices) - 1):
    #     updated = False
    #     for vertex in vertices:
    #         if dist[vertex] == float('inf'):
    #             continue
    #         for neighbor in graph.get_neighbors(vertex):
    #             weight = euclidean_dist(graph.get_position(vertex), graph.get_position(neighbor))

    #             if str(neighbor).endswith('0'):
    #                 weight = weight * -1

    #             if dist[neighbor] > dist[vertex] + weight:
    #                 dist[neighbor] = dist[vertex] + weight
    #                 prev[neighbor] = vertex
    #                 updated = True
    #     #early exit if there are no updates
    #     if not updated:
    #         break  
        
    # #fall back to point-to-point if no path exists
    # if dist[dest_node] == float('inf'):
    #     start_pos = graph.get_position(start_node)
    #     dest_pos = graph.get_position(dest_node)
    #     fly(start_pos, dest_pos)
    #     return [start_pos, dest_pos]

    # path = []
    # current_node = dest_node
    
    # while current_node is not None:
    #     path.append(graph.get_position(current_node))
    #     current_node = prev[current_node]

    # path.reverse()
    
    # for i in range(1, len(path)):
    #     fly(path[i-1], path[i])

    # return path

    return [] 

def dijkstra(start, dest):
    graph = MapGraph('graph.json')
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    vertices = graph.get_vertices() 
    dist = {node: float('inf') for node in vertices}
    dist[start_node] = 0
    prev_node = {node: None for node in vertices}
    pq = [(0, start_node)]  #minheap priority queue

    while pq:
        #first pop node w the smallest dist
        current_dist, current_node = heapq.heappop(pq)
        if current_node == dest_node:
            break
        if current_dist > dist[current_node]:
            continue
        for neighbor in graph.get_neighbors(current_node):
            weight = euclidean_dist(graph.get_position(current_node), graph.get_position(neighbor))
            new_dist = current_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev_node[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    path = []
    current_node = dest_node

    while current_node is not None:
        path.append(graph.get_position(current_node))
        current_node = prev_node[current_node]

    path.reverse()
    for i in range(1, len(path)):
        fly(path[i-1], path[i])

    return path

def astar(start, dest):
    graph = MapGraph('graph.json')
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    start_pos = graph.get_position(start_node)
    dest_pos = graph.get_position(dest_node)

    pq = []
    heapq.heappush(pq, (0, start_node)) 
    
    #g,f,and parent mappings
    g_costs = {start_node: 0}  
    f_costs = {start_node: euclidean_dist(start_pos, dest_pos)}  
    parents = {start_node: None}  #used later for rebuilding the path
    visited_nodes = set()

    while pq:
        _, current_node = heapq.heappop(pq)
        if current_node == dest_node:
            path = []
            while current_node is not None:
                path.append(graph.get_position(current_node))
                current_node = parents[current_node]
            path.reverse()

            for i in range(1, len(path)):
                fly(path[i - 1], path[i])

            return path

        visited_nodes.add(current_node)
        for neighbor in graph.get_neighbors(current_node):
            if neighbor in visited_nodes:
                continue

            tentative_g = g_costs[current_node] + euclidean_dist(graph.get_position(current_node), graph.get_position(neighbor))
            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                f_costs[neighbor] = g_costs[neighbor] + euclidean_dist(graph.get_position(neighbor), dest_pos)
                parents[neighbor] = current_node
                heapq.heappush(pq, (f_costs[neighbor], neighbor))
    return [] 

def min_spanning_tree(start, dest):
    graph = MapGraph('graph.json')
    start_node = graph.find_closest_vertex(start)
    dest_node = graph.find_closest_vertex(dest)
    pq = []
    src = start_node
    key = [float('inf')] * len(graph.get_vertices())
    parent = {v: None for v in graph.get_vertices()}
    in_mst = {v: False for v in graph.get_vertices()}
    
    heapq.heappush(pq, (0, src))
    key = {v: float('inf') for v in graph.get_vertices()}
    key[src] = 0

    while pq:
        u = heapq.heappop(pq)[1]
        if in_mst[u]:
            continue
        in_mst[u] = True
        
        for neighbor in graph.get_neighbors(u):
            weight = euclidean_dist(graph.get_position(u), graph.get_position(neighbor))
            if not in_mst[neighbor] and key[neighbor] > weight:
                key[neighbor] = weight
                heapq.heappush(pq, (key[neighbor], neighbor))
                parent[neighbor] = u
    
    current = dest_node
    path = []

    while current is not None:
        path.append(graph.get_position(current))
        current = parent[current]

    path.reverse()   
    for i in range(1, len(path)):
        fly(path[i - 1], path[i])

    return path

def search(algorithm, start, dest):
    if (algorithm == 'p2p'):
        return point_to_point(start, dest)
    elif(algorithm == 'fly'):
        return fly(start, dest)
    elif(algorithm == 'random'):
        return random_graph(start, dest)
    elif(algorithm == 'bfs'):
        return breadth_first(start, dest)
    elif(algorithm == 'bfsh'):
        return breadth_first_hub(start, dest)
    elif(algorithm == 'dfs'):
        return depth_first(start, dest)
    elif(algorithm == 'dfsb'):
        return depth_first_best(start, dest)
    elif(algorithm == 'bf'):
        return bellman_ford(start, dest)
    elif(algorithm == 'bfn'):
        return bellman_ford_negative(start, dest)
    elif(algorithm == 'dijkstra'):
        return dijkstra(start, dest)
    elif(algorithm == 'astar'):
        return astar(start, dest)
    elif(algorithm == 'mst'):
        return min_spanning_tree(start, dest)

    return []

if __name__ == "__main__":
    pass