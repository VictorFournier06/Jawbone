#imports
import numpy as np
import networkx as nx

#TSP approximation 1
#find a local minimum order than minimizes the sum of the distances between consecutive sentences (order 1) in O(N^2)
def insertion_sort(distances, permutation = []):
    permutation = list(permutation)
    #if no permutation is given, we create the default order
    if len(permutation) == 0:
        permutation = list(range(len(distances)))
    ordered = []
    for i in permutation:
        #find the position that minimizes the distance between the last sentence and the next sentence
        min = 2
        i_min = 0
        n = len(ordered)
        for j in range(n):
            if distances[i][j]+distances[i][(j+1)%n] < min: #we add the modulo to make the list circular, to avoid border bias, but we neglect border effects
                min = distances[i][j]+distances[i][(j+1)%n]
                i_min = (j+1)%n
        #insert the current sentence in the ordered list
        ordered.insert(i_min, i)
    return(ordered)

#TSP approximation 2
#alternative algorithm that tries to find a sentence to place better from in O(n^2) - bad algorithm
def greedy_sort(distances, permutation = []):
    permutation = list(permutation)
    #if no permutation is given, we create the default order
    if len(permutation) == 0:
        permutation = list(range(len(distances)))
    #for each sentence in the current permutation we try to find a better position
    n = len(permutation)
    for i in range(n):
        current = permutation[i] #the sentence we are trying to place
        #find the position that minimizes the distance between the last sentence and the next sentence
        min = 2
        i_min = 0
        for j in range(n): #the position we are trying to place the sentence
            if distances[current][permutation[j]]+distances[current][permutation[(j+1)%n]] < min: #we add the modulo to make the list circular, to avoid border bias, but we neglect border effects
                min = distances[current][permutation[j]]+distances[current][permutation[(j+1)%n]]
                i_min = (j+1)%n
        #if we found a better position, we swap the sentences
        if i_min != current:
            permutation.remove(current)
            permutation.insert(i_min, current)
    return(permutation)

#TSP approximation 3 (sort - transitivity hypothesis)
#pairwise_order is a matrix of pairwise orders (0 if i<=j, 1 if i>j)
def bubble_sort(pairwise_order, permutation = []):
    permutation = list(permutation)
    #if no permutation is given, we create the default order
    if len(permutation) == 0:
        permutation = list(range(len(pairwise_order)))
    for i in range(len(permutation)):
        for j in range(len(permutation)-1):
            if pairwise_order[permutation[j]][permutation[j+1]]:
                permutation[j], permutation[j+1] = permutation[j+1], permutation[j]
    return(permutation)

#TSP approximation 4 (sort - transitivity hypothesis)
def merge_sort(pairwise_order, permutation = []):
    permutation = list(permutation)
    #if no permutation is given, we create the default order
    if len(permutation) == 0:
        permutation = list(range(len(pairwise_order)))
    if len(permutation) > 1:
        mid = len(permutation)//2
        left = permutation[:mid]
        right = permutation[mid:]
        merge_sort(pairwise_order, left)
        merge_sort(pairwise_order, right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if pairwise_order[left[i]][right[j]]:
                permutation[k] = left[i]
                i += 1
            else:
                permutation[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            permutation[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            permutation[k] = right[j]
            j += 1
            k += 1
    return(permutation)

#TSP approximation 4 (sort - transitivity hypothesis)
def quick_sort(pairwise_order, permutation = []):
    permutation = list(permutation)
    #if no permutation is given, we create the default order
    if len(permutation) == 0:
        permutation = list(range(len(pairwise_order)))
    quick_sort_helper(pairwise_order, permutation, 0, len(permutation)-1)
    return(permutation)

def quick_sort_helper(pairwise_order, permutation, first, last):
    if first < last:
        splitpoint = partition(pairwise_order, permutation, first, last)
        quick_sort_helper(pairwise_order, permutation, first, splitpoint-1)
        quick_sort_helper(pairwise_order, permutation, splitpoint+1, last)
        
def partition(pairwise_order, permutation, first, last):
    pivotvalue = permutation[first]
    leftmark = first+1
    rightmark = last
    done = False
    while not done:
        while leftmark <= rightmark and pairwise_order[permutation[leftmark]][pivotvalue]:
            leftmark += 1
        while pairwise_order[pivotvalue][permutation[rightmark]] and rightmark >= leftmark:
            rightmark -= 1
        if rightmark < leftmark:
            done = True
        else:
            permutation[leftmark], permutation[rightmark] = permutation[rightmark], permutation[leftmark]
    permutation[first], permutation[rightmark] = permutation[rightmark], permutation[first]
    return(rightmark)

#function taking predictions of pairwise orderings probabilities and returning the order while enforcing transitivity
#Weighted transitivity closure, weights are the inverse logit of the probabilities
def order_from_pairwise(pairwise_probs):
    n = len(pairwise_probs)
    #create the graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            val = abs(np.log(pairwise_probs[i][j]/(1-pairwise_probs[i][j]))) 
            weights = 1/val if val >= 1e-7 else float('inf')
            if pairwise_probs[i][j] > 0.5:
                G.add_edge(i, j, weight=weights)
            else:
                G.add_edge(j, i, weight=weights)
    #compute the min weight transitive closure with Floyd-Warshall algorithm
    min_closure = min_weight_transitive_closure(G, weight='weight')

    return min_closure

def topological_sort(G):
    #use topological sorting to get the minimal order
    min_order = list(nx.topological_sort(G))
    return(min_order)    

def min_weight_transitive_closure(graph, weight='weight'):
    # Compute the minimum distances between each pair of nodes using Floyd-Warshall (the smaller the distance the more likely the ordering)
    min_distances = nx.floyd_warshall(graph, weight=weight)

    # Make a list of edges sorted by increasing weight
    edges = sorted(graph.edges(data=True), key=lambda t: t[2].get(weight, 1))

    # Create a new Directed Graph to represent the minimal weight transitive closure
    min_closure = nx.DiGraph()

    # Copy nodes from the original graph to min_closure
    min_closure.add_nodes_from(graph.nodes)

    # Add edges based on the Floyd-Warshall results, in increasing weight order, ensuring acyclicity
    for u, v, _ in edges:
        if u != v:
            if min_distances[u][v] != float('inf') and min_distances[u][v] <= min_distances[v][u]:
                # Add an edge only if it doesn't create a cycle
                if not nx.has_path(min_closure, v, u):
                    min_closure.add_edge(u, v, weight=min_distances[u][v])
            elif min_distances[v][u] != float('inf') and min_distances[v][u] < min_distances[u][v]:
                # Add an edge only if it doesn't create a cycle
                if not nx.has_path(min_closure, u, v):
                    min_closure.add_edge(v, u, weight=min_distances[v][u])
    return min_closure