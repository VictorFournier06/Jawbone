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
