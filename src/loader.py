#imports
import os
from os.path import sep

def get_chunks(text, max_token_size = 256, tokenizer = None):
    #split the text into chunks of max_token_size tokens
    words = text.split()
    chunks = []
    chunk_size = 0
    chunk = ""
    for word in words:
        l = len(tokenizer.tokenize(word))
        if chunk_size + l < max_token_size:
            chunk_size += l
            chunk += word + " "
        else:
            chunks.append(chunk)
            chunk_size = l
            chunk = word + " "
    chunks.append(chunk)
    return chunks

def load_books(db_type = "Tom Swift", mode = "sentences", max_token_size = 256, tokenizer = None):
    #Open all the files in the directory and store them in the list 'files'
    files = []
    for filename in os.listdir(os.path.dirname(os.getcwd()) + sep + "Books" + sep + db_type):
        if filename.endswith(".txt"):
            with open(os.path.dirname(os.getcwd()) + sep + "Books" + sep + db_type + sep + filename, 'r') as f:
                text = f.read()
                files.append(text)

    #get rid of the metadata at the beginning and end of each file
    for i in range(len(files)):
        files[i] = files[i].split("***")[2]
        files[i] = files[i].split("End of the Project Gutenberg EBook")[0]
        
    #split each file into a list of paragraphs or a list of sentences
    if mode == "paragraphs":
        sentences = [0 for i in range(len(files))]
        for i in range(len(files)):
            sentences[i] = files[i].split("\n\n")
    elif mode == "sentences":
        sentences = [0 for i in range(len(files))]
        for i in range(len(files)):
            sentences[i] = files[i].split("\n")
    elif mode == "chunks": #batch sentences based on maxmimum number of tokens
        sentences = [0 for i in range(len(files))]
        for i in range(len(files)):
            sentences[i] = get_chunks(files[i], max_token_size = max_token_size, tokenizer = tokenizer)
    else:
        print("Invalid mode")
        return

    #get rid of paragraphs and sentences with less than 10 characters
    for i in range(len(files)):
        sentences[i] = [x for x in sentences[i] if len(x.replace(' ', '')) > 10]
    
    return(sentences)