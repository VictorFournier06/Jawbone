#imports
import os

def load_books(db_type = "Tom Swift"):
    #Open all the files in the directory and store them in the list 'files'
    files = []
    for filename in os.listdir(os.path.dirname(os.getcwd()) + "\\Books\\" + db_type):
        if filename.endswith(".txt"):
            with open(os.path.dirname(os.getcwd()) + "\\Books\\" + db_type + "\\" + filename, 'r') as f:
                text = f.read()
                files.append(text)

    #get rid of the metadata at the beginning and end of each file
    for i in range(len(files)):
        files[i] = files[i].split("***")[2]
        files[i] = files[i].split("End of the Project Gutenberg EBook")[0]
        
    #split each file into a list of paragraphs or a list of sentences
    paragraphs = [0 for i in range(len(files))]
    sentences = [0 for i in range(len(files))]
    for i in range(len(files)):
        paragraphs[i] = files[i].split("\n\n")
        sentences[i] = files[i].split("\n")

    #batch sentences based on maxmimum number of tokens

    #get rid of paragraphs and sentences with less than 10 characters
    for i in range(len(files)):
        paragraphs[i] = [x for x in paragraphs[i] if len(x.replace(' ', '')) > 10]
        sentences[i] = [x for x in sentences[i] if len(x.replace(' ', '')) > 2]
    
    return(paragraphs, sentences)