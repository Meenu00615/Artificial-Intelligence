#!/usr/bin/env python
# coding: utf-8

# In[ ]:


AI_3rd_Mine side_ with graph


# In[16]:


import networkx as nx
import matplotlib.pyplot as plt
import queue as Q



def getPriorityQueue(list):
	q = Q.PriorityQueue()
	for node in list:
		q.put(Ordered_Node(heuristics[node],node))
	return q,len(list)



def BFSUtil(G, v, visited, final_path, dest, goal):
	if goal == 1:
		return goal
	visited[v] = True
	final_path.append(v)
	if v == dest:
		goal = 1
	else:
		pq_list = []
		pq,size = getPriorityQueue(G[v])
		for i in range(size):
			pq_list.append(pq.get().description)
		for i in pq_list:
			if goal != 1:
				#print "current city:", i
				if visited[i] == False :
					goal = BFSUtil(G, i, visited, final_path, dest, goal)
	return goal

def BFS(G, source, dest, heuristics, pos):
    visited = {}
    for node in G.nodes():
        visited[node] = False
    final_path = []
    goal = BFSUtil(G, source, visited, final_path, dest, 0)
    prev = -1
    total_cost = 0  # Initialize total cost to zero
    for i in range(len(final_path) - 1):  # Iterate over the nodes in the final path
        u = final_path[i]
        v = final_path[i + 1]
        edge_data = G.get_edge_data(u, v)
        if edge_data:  # Check if the edge exists
            total_cost += int(edge_data['length'])  # Increment total cost by the length of the edge
    print("Total cost:", total_cost)  # Print the total cost
    for var in final_path:
        if prev != -1:
            curr = var
            nx.draw_networkx_edges(G, pos, edgelist=[(prev, curr)], width=2.5, alpha=0.8, edge_color='black')
            prev = curr
        else:
            prev = var
    return total_cost


class Ordered_Node(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description

    def __lt__(self, other):
        return self.priority < other.priority


def getHeuristics(G):
	heuristics = {}
	f = open('heuristics.txt')
	for i in G.nodes():
		node_heuristic_val = f.readline().split()
		heuristics[node_heuristic_val[0]] = node_heuristic_val[1]
	return heuristics



#takes input from the file and creates a weighted graph
def CreateGraph():
	G = nx.Graph()
	f = open('input.txt')
	n = int(f.readline())
	for i in range(n):
		graph_edge_list = f.readline().split()
		G.add_edge(graph_edge_list[0], graph_edge_list[1], length = graph_edge_list[2])
	source, dest= f.read().splitlines()
	return G, source, dest



def DrawPath(G, source, dest):
	pos = nx.spring_layout(G)
	val_map = {}
	val_map[source] = 'green'
	val_map[dest] = 'red'
	values = [val_map.get(node, 'blue') for node in G.nodes()]
	nx.draw(G, pos, with_labels = True, node_color = values, edge_color = 'b' ,width = 1, alpha = 0.7)  #with_labels=true is to show the node number in the output graph
	edge_labels = dict([((u, v,), d['length']) for u, v, d in G.edges(data = True)])
	nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, label_pos = 0.5, font_size = 11) #prints weight on all the edges
	return pos

if __name__ == "__main__":
    G, source, dest = CreateGraph()
    heuristics = getHeuristics(G)
    pos = DrawPath(G, source, dest)
    total_cost = BFS(G, source, dest, heuristics, pos)
    print("Total cost:", total_cost)
    plt.show()


# # AI_1st
# AIM: Write a python program to implement a Blind Search (BFS) program in AI.

# In[13]:


from collections import deque
import time
import networkx as nx
import matplotlib.pyplot as plt

def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])
    start_time = time.time()
    while queue:
        current, path = queue.popleft()
        if current == goal:
            end_time = time.time()
            elapsed_time = end_time - start_time
            return path, elapsed_time

        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                queue.append((neighbor, path + [neighbor]))

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'G'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C'],
    'G': ['C', 'H'],
    'H': ['G']
}

start_node = 'B'
goal_node = 'H'

result, time_bfs = bfs(graph, start_node, goal_node)
print(f"Path from {start_node} to {goal_node}: {result} , Time: {time_bfs} seconds")

# Plot the graph with the path
G = nx.Graph(graph)
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10)

# Highlight the path
edges = [(result[i], result[i+1]) for i in range(len(result)-1)]
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)

plt.show()


# # AI_2nd
# AIM: Write a Python Program to Implement Uninformed Searches( Depth first Search )

# In[12]:


from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import time

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def plot(self):
        G = nx.DiGraph()
        for u, neighbors in self.graph.items():
            for v in neighbors:
                G.add_edge(u, v)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10)
        plt.show()

    def dfs_stack(self, start, destination):
        visited = defaultdict(bool)
        stack = [start]
        traversal_order = []

        start_time = time.time()

        while stack:
            vertex = stack.pop()
            if not visited[vertex]:
                traversal_order.append(vertex)
                visited[vertex] = True
                if vertex == destination:
                    break
                print("Visited:", vertex)
                print("Stack:", stack)
                for neighbor in reversed(self.graph[vertex]):
                    if not visited[neighbor]:
                        stack.append(neighbor)

        end_time = time.time()

        print("Traversal order:", traversal_order)
        print("Time taken:", end_time - start_time, "seconds")

# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
g.add_edge(1, 4)
g.add_edge(4, 5)
g.add_edge(4, 6)
g.add_edge(5, 6)

start_node = 2
destination_node = 6
print("DFS from node", start_node, "to", destination_node)
g.dfs_stack(start_node, destination_node)

# Plot the graph
g.plot()


# # AI_3rd
# AIM: Write a python program to implement Greedy Search in AI.

# In[11]:


import heapq

class Node:
    def __init__(self, state, cost, heuristic, parent=None):
        self.state = state
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class Graph:
    def __init__(self):
        self.edges = {}  # Initialize an empty dictionary to store edges

    def add_edge(self, node, neighbor, cost):
        if node not in self.edges:
            self.edges[node] = []
        self.edges[node].append((neighbor, cost))


def best_first_search(initial_state, goal_state, heuristic_function, successors_function):
    priority_queue = []
    heapq.heappush(priority_queue, Node(initial_state, 0, heuristic_function(initial_state)))

    explored = set()

    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        explored.add(current_node.state)

        if current_node.state == goal_state:
            return retrace_path(current_node)

        for successor_state, cost in successors_function(current_node.state):
            if successor_state not in explored:
                successor_node = Node(successor_state, current_node.cost + cost, heuristic_function(successor_state),
                                      parent=current_node)
                heapq.heappush(priority_queue, successor_node)

    return None

def retrace_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()

    return path


def heuristic(state):
    return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])

def successors(state):
    if state not in graph.edges:
        return []
    return graph.edges[state]

heuristic_costs = {
    'Arad': 342,
    'Pitesi': 10,
    'Fagaras': 176,
    'Timisoara': 329,
    'Sibiu': 253,
    'Rimnicu': 193,
    'Mehadia': 241,
    'Dobreta': 242,
    'Cralova': 160,
    'Bucharest': 0,
    'Lugo': 100,
}

graph = Graph()
graph.add_edge('Arad', 'Timisoara', 118)
graph.add_edge('Arad', 'Sibiu', 140)
graph.add_edge('Sibiu', 'Rimnicu', 80)
graph.add_edge('Sibiu', 'Fagaras', 99)
graph.add_edge('Fagaras', 'Bucharest', 211)
graph.add_edge('Timisoara', 'Lugo', 111)
graph.add_edge('Lugo', 'Mehadia', 70)
graph.add_edge('Mehadia', 'Dobreta', 75)
graph.add_edge('Dobreta', 'Cralova', 120)
graph.add_edge('Cralova', 'Rimnicu', 146)
graph.add_edge('Cralova', 'Pitesi', 138)
graph.add_edge('Pitesi', 'Bucharest', 101)
graph.add_edge('Pitesi', 'Rimnicu', 97)

initial_state = 'Arad'
goal_state = 'Bucharest'

path = best_first_search(initial_state, goal_state, lambda x: heuristic_costs[x], successors)

if path:
    print("Goal reached!")
    print("Path:", path)
    total_cost = 0
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i + 1]
        for neighbor, cost in graph.edges[current_state]:
            if neighbor == next_state:
                total_cost += cost
                break

    print("Total Path Cost:", total_cost)
else:
    print("Goal not reached.")


# In[8]:


import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(graph, path):
    G = nx.Graph()
    for node, neighbors in graph.edges.items():
        for neighbor, cost in neighbors:
            G.add_edge(node, neighbor, weight=cost)
    
    pos = nx.spring_layout(G)  # Positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw labels
    nx.draw_networkx_labels(G, pos)

    # Highlight path
    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)

    # Display the graph
    plt.axis('off')
    plt.show()

# Plot the graph with the path
plot_graph(graph, path)


# # AI_4th
# AIM: Write a python program to implement Pure Heuristic : A* Search in AI.

# In[21]:


import networkx as nx
import matplotlib.pyplot as plt
import heapq

def heuristic(n):
    H_dist = {
        'A': 10,
        'B': 6,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }
    return H_dist.get(n, float('inf'))  # Return infinity if the node is not found in the heuristic dictionary


# Define the A* algorithm
def aStarAlgo(start, goal):
    open_list = []
    closed_set = set()
    came_from = {}
    g_score = {node: float('inf') for node in Graph_nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in Graph_nodes}
    f_score[start] = heuristic(start)
    heapq.heappush(open_list, (f_score[start], start))

    while open_list:
        current_f, current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current_node)

        for neighbor, weight in Graph_nodes[current_node]:
            tentative_g_score = g_score[current_node] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                if neighbor not in closed_set:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None

# Define the graph nodes and edges
Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('A', 6), ('C', 3), ('D', 2)],
    'F': [('A', 3), ('G', 1), ('H', 7)],
    'C': [('B', 3), ('D', 1), ('E', 5)],
    'D': [('B', 2), ('C', 1), ('E', 8)],
    'E': [('C', 5), ('D', 8), ('I', 5), ('J', 5)],
    'G': [('F', 1), ('I', 3)],
    'H': [('F', 7), ('I', 2)],
    'I': [('E', 5), ('G', 3), ('H', 2), ('J', 3)],
    'J': [('E', 5), ('I', 3)]  # Add node 'J' with its connections
}

# Call the A* algorithm to find the path
path = aStarAlgo('A','J')

# Create a directed graph
G = nx.Graph()

# Add nodes and edges to the graph
for node, neighbors in Graph_nodes.items():
    G.add_node(node)
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# Visualize the graph with the path highlighted
pos = nx.spring_layout(G)

# Draw nodes and edges without highlighting the path
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
edge_labels = {(n1, n2): d['weight'] for n1, n2, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Highlight the path
path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.0)

# Display the graph
plt.title('Graph Visualization with A* Path')
plt.show()


# # AI_5th
# AIM: Write a Program to Implement the Travelling Salesman Problem using Python. THEORY:

# In[22]:


from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def dfs(graph, start, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]
    visited.add(start)
    if len(visited) == len(graph):
        path.append(start)
        return path
    for neighbor, _ in sorted(graph[start], key=lambda x: x[1]):
        if neighbor not in visited:
            path.append(neighbor)
            new_path = dfs(graph, neighbor, visited.copy(), path)
            if new_path:
                return new_path
            path.pop()

def bfs(graph, start):
    queue = [(start, [start])]
    while queue:
        (node, path) = queue.pop(0)
        for neighbor, _ in sorted(graph[node], key=lambda x: x[1]):
            if neighbor not in path:
                if len(path) == len(graph) - 1:
                    return path + [neighbor, start]
                else:
                    queue.append((neighbor, path + [neighbor]))

def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        for neighbor, weight in graph[current_city]:
            if neighbor == next_city:
                cost += weight
                break
    return cost

def visualize_graph(graph):
    G = nx.Graph()

    #Adding nodes
    for node in graph:
        G.add_node(node)

    #adding edges
    for node, edges in graph.items():
        for edge, weight in edges:
            G.add_edge(node, edge, weight=weight)

    #graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue', edge_color='black', width=1, alpha=0.7)

    #Edge labels
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Graph Representation of Cities and Distances")
    plt.show()

if __name__ == "__main__":
    graph_nodes = {
        'ARAD': [('ARAD',0), ('TIMISORA',235), ('LUGOJ',256), ('MEHADIA',248), ('DOBRETA',277), ('CAIOVA',158), ('PITESTI',180)],
        'TIMISORA': [('ARAD',135), ('TIMISORA',0), ('LUGOJ',260), ('MEHADIA',233), ('DOBRETA',294), ('CAIOVA',198), ('PITESTI',258)],
        'LUGOJ': [('ARAD',140), ('TIMISORA',280), ('LUGOJ',0), ('MEHADIA',271), ('DOBRETA',273), ('CAIOVA',259), ('PITESTI',266)],
        'MEHADIA': [('ARAD',200), ('TIMISORA',298), ('LUGOJ',283), ('MEHADIA',0), ('DOBRETA',263), ('CAIOVA',218), ('PITESTI',220)],
        'DOBRETA': [('ARAD',212), ('TIMISORA',275), ('LUGOJ',229), ('MEHADIA',214), ('DOBRETA',0), ('CAIOVA',274), ('PITESTI',286)],
        'CAIOVA': [('ARAD',148), ('TIMISORA',271), ('LUGOJ',254), ('MEHADIA',286), ('DOBRETA',269), ('CAIOVA',0), ('PITESTI',298)],
        'PITESTI': [('ARAD',214), ('TIMISORA',275), ('LUGOJ',214), ('MEHADIA',272), ('DOBRETA',281), ('CAIOVA',270), ('PITESTI',0)],
    }

    start_city = 'ARAD'

    print("DFS Path:", dfs(graph_nodes, start_city))
    print("BFS Path:", bfs(graph_nodes, start_city))

    visualize_graph(graph_nodes)

    path = dfs(graph_nodes, start_city)
    cost = calculate_path_cost(graph_nodes, path)
    print("Path cost (DFS):", cost)

    path = bfs(graph_nodes, start_city)
    cost = calculate_path_cost(graph_nodes, path)
    print("Path cost (BFS):", cost)

