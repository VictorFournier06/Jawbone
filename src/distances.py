from sentence_transformers import util

#compute the semantic distance between each pair of sentences in an embedding
def pairwise_dist(sentence_embedding):
    return([1-util.cos_sim(sentence_embedding[i], sentence_embedding[i]) for i in range(len(sentence_embedding))])

#compute the average distance between consecutive sentences (order 1)
def avg_consecutive_dist(order, pairwise_dist): #distances are the distances for 1 book
    if len(pairwise_dist) != len(order):
        raise Exception("inputs are not of same size")
    sum = 0
    n = len(order)
    for i in range(n):
        sum += pairwise_dist[order[i]][order[(i+1)%n]]
    return(sum/n)

#permutation swap distance
def kendall_tau(permutation1, permutation2):
    if len(permutation1) != len(permutation2):
        raise Exception("permutations are not of the same length")
    n = len(permutation1)
    sum = 0
    for i in range(n):
        for j in range(i+1, n):
            if (permutation1[i] < permutation1[j]) != (permutation2[i] < permutation2[j]):
                sum += 1
    return(1-4*sum/(n-1)/n)

#R distance - unidirectional adjacency distance
def avg_R_dist(permutation1, permutation2):
    if len(permutation1) != len(permutation2):
        raise Exception("permutations are not of the same length")
    n = len(permutation1)
    pairs1 = []
    pairs2 = []
    for i in range(n):
        pairs1.append((permutation1[i], permutation1[(i+1)%n]))
        pairs2.append((permutation2[i], permutation2[(i+1)%n]))
    #return the number of common pairs
    return(1-len(set(pairs1).intersection(set(pairs2)))/n)