# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:17:43 2023

@author: SHOUMYA
"""

import heapq


graph = {'Oradea':{'Zerind':71,'Sibiu':151},'Zerind':{'Oradea':71,'Arad':75},
          'Arad':{'Zerind':75,'Timisoara':118,'Sibiu':140},
          'Timisoara':{'Arad':118,'Lugoj':111},
          'Lugoj':{'Timisoara':111,'Mehadia':70},
          'Mehadia':{'Lugoj':70,'Dobreta':70},
          'Dobreta':{'Mehadia':75,'Craiova':120},
          'Craiova':{'Dobreta':120,'Rimnicu Vilcea':146,'Pitesti':138},
          'Sibiu':{'Arad':140,'Oradea':151,'Rimnicu Vilcea':80,'Fagaras':99},
          'Rimnicu Vilcea':{'Sibiu':80,'Craiova':146,'Pitesti':97},
          'Fagaras':{'Sibiu':99,'Bucharest':221},
          'Pitesti':{'Bucharest':101,'Rimnicu Vilcea':97,'Craiova':138},
          'Bucharest':{'Fagaras':211,'Pitesti':101,'Giurgiu':90,'Urziceni':85},
          'Giurgiu':{'Bucharest':90},
          'Urziceni':{'Bucharest':85,'Hirsova':98,'Vaslui':142},
          'Hirsova':{'Urziceni':98,'Eforie':86},
          'Eforie':{'Hirsova':86},
          'Vaslui':{'Urziceni':142,'Iasi':92},
          'Iasi':{'Neamt':87,'Vaslui':92},
          'Neamt':{'Iasi':87}}
hsld = {'Arad':366, 'Bucharest':0, 'Craiova':160, 'Dobreta':242, 'Eforie':161, 'Fagaras':176, 'Giurgiu':77, 'Hirsova':151, 'Iasi':226, 'Lugoj':244, 'Mehadia':241, 'Neamt':234, 'Oradea':380, 'Pitesti':100, 'Rimnicu Vilcea':193, 'Sibiu':253, 'Timisoara':329, 'Urziceni':80, 'Vaslui':199, 'Zerind':374}

def heuristic(name):
        return hsld[name]

start = 'Arad'
goal = 'Bucharest'


# Create the open and closed lists
open_list = []
heapq.heapify(open_list)
closed_list = set()

# Set the start node
h = heuristic(start)
f = h
current_node = (f, start, 0, ())
heapq.heappush(open_list, current_node)
flag = 0

# Loop until you find the end
while len(open_list) > 0:

    # Get the current node
    current_node = heapq.heappop(open_list)
    current_f, current_vertex, current_cost, current_path = current_node

    # If we have found the goal node, print the path
    if current_vertex == goal:
        current_path = current_path + (current_vertex,)
        print(current_cost)
        print(current_path)
        flag = 1
        break

    # Mark the current node as closed
    closed_list.add(current_vertex)

    # Get all the neighbors
    neighbors = graph[current_vertex].keys()

    # Loop through all the neighbors
    for neighbor in neighbors:
        if neighbor not in closed_list:
            # Calculate the cost to the neighbor
            cost = graph[current_vertex][neighbor]
            total_cost = current_cost + cost
            h = heuristic(neighbor)
            f = h

            # Create a neighbor node with the total cost
            neighbor_node = (f, neighbor, total_cost, current_path + (current_vertex,))

            # If the neighbor is not in the open list, add it
            if neighbor not in [i[1] for i in open_list]:
                heapq.heappush(open_list, neighbor_node)
            else:
                # If the neighbor is in the open list, check if it has a lower cost
                for i, v in enumerate(open_list):
                    if v[1] == neighbor:
                        if neighbor_node[2] < v[2]:
                            open_list[i] = neighbor_node
                            heapq.heapify(open_list)
                            break

if flag == 0:
    print("No path found!")