#find a local minimum order than minimizes the sum of the distances between consecutive sentences (order 1) in O(N^2)
def greedy_add(distances, permutation = []):
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


#alternative algorithm that tries to find a sentence to place better from in O(n^2)
def greedy_sort(distances, permutation = []):
    permutation = list(permutation)
    #if no permutation is given, we start create the default order
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