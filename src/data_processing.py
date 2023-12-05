#imports
import numpy as np
import torch
import networkx as nx

#function to split the sentences into discard, training, validation and testing sets while keeping the order
def split_sentences(sentence_embedding, fraction_to_keep = 0.1, fraction_to_train = 0.8, fraction_to_test = 0.5):
    sentences_train, sentences_val, sentences_test = [], [], []
    for book in range(len(sentence_embedding)):
        sentences = torch.tensor(sentence_embedding[book]).to("cuda")                                   #convert to tensor
        num_elements_to_keep = int(sentences.size(0)*fraction_to_keep)                                  #number of elements to keep
        num_elements_to_train = int(num_elements_to_keep*fraction_to_train)                             #number of elements to train
        num_elements_to_test = int((num_elements_to_keep-num_elements_to_train)*fraction_to_test)       #number of elements to test
        mask_keep = torch.randperm(sentences.size(0)) < num_elements_to_keep                            #mask to keep the elements
        mask_train = torch.randperm(num_elements_to_keep) < num_elements_to_train                       #mask to train the elements
        mask_test = torch.randperm((num_elements_to_keep-num_elements_to_train)) < num_elements_to_test #mask to test the elements
        sentences_keep = sentences[mask_keep]                                                       
        sentences_train.append(sentences_keep.clone()[mask_train])
        sentences_val.append(sentences_keep[~mask_train].clone()[~mask_test])
        sentences_test.append(sentences_keep[~mask_train][mask_test])
    return sentences_train, sentences_val, sentences_test

#function to create the database of the pairs of a subset of sentences
def create_database(sentence_embeddings):
    #create the database
    X = []
    y = []
    
    for i in range(len(sentence_embeddings)):               #for each book
        X_i, y_i = create_pairs(sentence_embeddings[i])
        X.append(X_i)
        y.append(y_i)
    
    return torch.cat(X), torch.cat(y)

def create_pairs(sentence_embeddings):
    if not(torch.is_tensor(sentence_embeddings) or sentence_embeddings.is_cuda):
        sentence_embeddings = torch.tensor(sentence_embeddings).to("cuda")
    #create the database
    X = []
    y = []
    
    for j in range(sentence_embeddings.size(0)):        #for each ordered pair of sentences
        for k in range(sentence_embeddings.size(0)):
            if j != k:
                # Concatenate sentence embeddings
                input_pair = torch.cat((sentence_embeddings[j], sentence_embeddings[k]), dim=0).to("cuda")
                X.append(input_pair)
                # Label whether the first sentence comes before the second
                y.append(torch.tensor(j < k, dtype=torch.float).to("cuda"))
    
    return torch.stack(X), torch.stack(y).unsqueeze(1)

#add any diagonal values to the flattened matrix y_pred2
#convert y_pred2 from a list to a shifted antisymmetric matrix by averaging the predictions of the upper and lower triangular
def pred_to_pairwise(pred):
    n = np.ceil(pred.shape[0]**0.5).astype(int)
    L = pred.cpu().detach().numpy().flatten().tolist()
    if len(L < n*n) :
        for i in range(n):
            L.insert(i*i, 0.5)
    pairwise_probabilities = [[(L[i*n+j]+(1-L[j*n+i]))/2 for i in range(n)] for j in range(n)]
    return(pairwise_probabilities)

#function taking predictions of pairwise orderings probabilities and returning the order while enforcing transitivity
#Weighted transitivity closure, weights are the inverse logit of the probabilities
def weighted_transitivity_closure(pairwise_probs):
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
    
    #use topological sorting to get the minimal order
    min_order = list(nx.topological_sort(min_closure))
    return min_order

def min_weight_transitive_closure(graph, weight='weight'):
    # Compute the minimum distances between each pair of node using Floyd-Warshall (the smaller the distance the more likely the ordering)
    min_distances = nx.floyd_warshall(graph, weight=weight)

    # Create a new Directed Graph to represent the minimal weight transitive closure
    min_closure = nx.DiGraph()

    # Copy nodes from the original graph to min_closure
    min_closure.add_nodes_from(graph.nodes)

    # Add edges based on the Floyd-Warshall results, ensuring acyclicity
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v and min_distances[u][v] != float('inf') and min_distances[u][v] <= min_distances[v][u]:
                # Add an edge only if it doesn't create a cycle
                if not nx.has_path(min_closure, v, u):
                    min_closure.add_edge(u, v, weight=min_distances[u][v])

    return min_closure
    