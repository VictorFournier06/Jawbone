#imports
import numpy as np
import torch

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
            # Concatenate sentence embeddings
            input_pair = torch.cat((sentence_embeddings[j], sentence_embeddings[k]), dim=0).to("cuda")
            X.append(input_pair)
            # Label whether the first sentence comes before the second
            if j==k:
                y.append(torch.tensor(0.5).to("cuda"))
            else:
                y.append(torch.tensor(j < k, dtype=torch.float).to("cuda"))
            
    
    return torch.stack(X), torch.stack(y).unsqueeze(1)

#convert y2_pred from list to matrix
def flattened_to_matrix(pred):
    #Ouput matrix is of size n*n
    n = int(pred.shape[0]**0.5)
    if n != pred.shape[0]**0.5:
        print("size error, input is not a square matrix")
    L = pred.cpu().detach().numpy().flatten().tolist()
    
    '''
    #add diagonal values
    if len(L < n*n) :
        for i in range(n):
            L.insert(i*i, 0.5)
    '''

    #average predictions, diagonal values get set to 0.5
    pairwise_probabilities = [[L[i+n*j] for i in range(n)] for j in range(n)]
    return(np.array(pairwise_probabilities))

#convert M to a shifted antisymmetric matrix by averaging the predictions of the upper and lower triangular
def average_matrix(M):
    n = M.shape[0]
    M2 = M.copy()
    for i in range(n):
        for j in range(n):
            M2[i][j] = (M[i][j]+(1-M[j][i]))/2
    return(M2)

#convert a transitive graph of directed edges to a prediction-like list of pairwise orderings
def edges_to_pred(G):
    L = []
    for i in range(len(G)):
        for j in range(len(G)):
            if i != j:
                L.append(G.has_edge(i,j))
    return(np.array(L).astype(int))