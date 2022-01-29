#NIKOS KARIOTIS 3243

import math
import sys

out = open("out.txt" , "r") #open the file that was written by indexing.py
nodes = []
graph = []
parentsMapD = {}
parentsMapA = {}
iterations = 0
iterations1 = 0

for l in out: #fill an array with the lines of the out.txt file
    nodes.append(l)
    
def create_graph(nodes): #reconstructs the storage structure 
    global graph
    for i in range(len(nodes)):
        tmp = nodes[i].split()
        neighbours = []
        distance = []
        temp = []
        x = float(tmp[1])
        y = float(tmp[2])
        coords = [x,y]
        for j in range(3,len(tmp),2):
            neighbour = int(tmp[j])
            neighbours.append(neighbour)
        for j in range(4,len(tmp),2):
            dist = float(tmp[j])
            distance.append(dist)
        for j in range(len(neighbours)):
            temp.append([neighbours[j],distance[j]])
        graph.append([coords,temp])
   
def estimate_heuristic_cost(current,target): #it returns the euclidean distance of current node and target node
    global graph
    coords1 = graph[current][0] #take the coords of the current from the storage structure
    current_x = coords1[0]
    current_y = coords1[1]
    coords2 = graph[target][0] #same for the target
    target_x = coords2[0]
    target_y = coords2[1]
    euclidean_distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2) #calculate euclidean
    return euclidean_distance

class PriorityQueue(object): #hand-made minimum Priority queue
    def __init__(self):
        self.queue = []

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
  
    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)
  
    # for popping an element based on Priority
    def pop(self):
        try:
            minimum = self.queue[0][1]
            index = 0
            for i in range(len(self.queue)):
                if(self.queue[i][1] < minimum):
                    minimum = self.queue[i][1]
                    index = i
            item = self.queue[index][0]
            del self.queue[index]
            return item
        except IndexError:
            print()
            exit()

            
def dijkstra(source,target):
    global graph,iterations
    queue = PriorityQueue()
    visited = set() #visited nodes
    parentsMapD = {} #concentrate the path
    distances = {vertex: float('infinity') for vertex in range(len(graph))}
    distances[source] = 0
    queue.insert([source,0])
    while(not queue.isEmpty()):
        node = queue.pop()
        
        if(node not in visited):
            visited.add(node)
            iterations += 1

        if(node == target): #return of the function
            break
       
        for element in graph[node][1]: #check every neighbour of the node
            adjNode = element[0]
            weight = element[1]
            if (adjNode in visited): #if it is visited already continue with the next one
                continue
            newDistance = distances[node] + weight #if not calculate the cost of the path from source to neighbour
            if(distances[adjNode] > newDistance): #if that path is smaller than what we ve recorded, adjust
                parentsMapD[adjNode] = node
                distances[adjNode] = newDistance
                queue.insert([adjNode,newDistance]) 
		
    return parentsMapD,distances



def a_star(source,target):
    global graph,iterations1
    queue = PriorityQueue()
    visited = set()
    parentsMapA = {}
    g_score = {vertex: float('infinity') for vertex in range(len(graph))}
    g_score[source] = 0
    f_score = {vertex: float('infinity') for vertex in range(len(graph))} #f(n) = g(n) + h(n)
    euc = estimate_heuristic_cost(source,target)
    f_score[source] = euc #f(n) for source is equal to the euclidean distance from the target
    queue.insert([source,euc]) 
    while(not queue.isEmpty()):
        node = queue.pop()
        
        if(node not in visited):
            visited.add(node)
            iterations1 += 1
            
        if(node == target):
            break

        for element in graph[node][1]:
            adjNode = element[0]
            weight = element[1]
            if(adjNode in visited):
                continue
            newG = g_score[node] + weight # same as Diijkstra
            if(g_score[adjNode] > newG):
                parentsMapA[adjNode] = node
                g_score[adjNode] = newG
                euc = estimate_heuristic_cost(adjNode,target)
                newCost = euc + newG # f(n) = g(n) + h(n)
                f_score[adjNode] = newCost
                queue.insert([adjNode,newCost])
    return parentsMapA,f_score

def make_path(parent,source,target): #find the path from the dictionary with nodes that returned from diijkstra and A*
    v = target
    path = []
    while(v != source):
        path.append(v) #append to an array each node from the reversed path 
        v = parent[v] #we start from the target and then we go backwards until we find the source node
    path.append(source)
    return path[::-1] #return the reversed array so that it starts from source node


def calculate_distance(nodeCosts,target): #the total cost of the path
    return nodeCosts[target]

        
def main(argv):
    source = int(argv[1]) #first command line argument(source Node)
    target = int(argv[2]) #second command line argument(target Node)
    create_graph(nodes)
    parentsMapD, nodeCosts = dijkstra(source,target)
    path = make_path(parentsMapD,source,target)
    d_distance = calculate_distance(nodeCosts,target)
    print("\n" + "Dijkstra:")
    print("Shortest path length = " + str(len(path)))
    print("Shortest path distance = " + str(d_distance))
    print("Shortest path = " + str(make_path(parentsMapD,source,target)))
    print("Number of visited nodes = " + str(iterations))
    
    parentsMapA , node_costs = a_star(source,target)
    a_distance = calculate_distance(node_costs,target)
    path = make_path(parentsMapA,source,target)
    print("\n" + "Astar:")
    print("Shortest path length = " + str(len(path)))
    print("Shortest path distance = " + str(a_distance))
    print("Shortest path = " + str(make_path(parentsMapA,source,target)))
    print("Number of visited nodes = " + str(iterations1))
    
if __name__ == "__main__":
    main(sys.argv)  
