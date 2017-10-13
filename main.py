
#Open and load the 

import random; random.seed(123)
import string
import codecs


from nltk.stem.porter import PorterStemmer

#Partition file into separate paragraphs. Paragraphs are text chunks separated by empty line.
#Partitioning result should be a list of paragraphs. For us, each paragraph will be a separate document.


def paragrahp_file(file):
    """ Pre process file and return all paragraphs in an list"""
    text = []
    paragraph = ""
    for line in f:
        #continue if line.isspace()
        if(line.isspace()):
            if(paragraph != ""):
                text.append(paragraph)
            paragraph = ""
            continue
        paragraph += line
    return text


def tokenize_and_clean_text(t_file):
    """Tokenizs the text and remove paragrahs containing Gutenberg """
    #Empty array.
    t_p_text = []
    t_p_text_orig = []
    #Tokenize paragraphs into list of words

    #Table of characters to remove from our string
    table = str.maketrans(dict.fromkeys(string.punctuation+"\n\r\t"))

    for para in t_file:
        paragraph = []
        """ Remove (filter out) paragraphs containing the word “Gutenberg” (=headers and footers)"""
        if para.__contains__('Gutenberg') or para.__contains__('gutenberg'):
            continue
        #turn paragraph into lowercase letters.
        para_lower = para.lower()
        #clean the text of paragraphs and \n\r\t characters
        para_clean = para_lower.translate(table)
        #Add the list split into words to the list
        
        t_p_text_orig.append(para_clean)
        paragraph.append(para_clean.split())    

        #Appen this
        t_p_text.append(paragraph)
    
    return t_p_text, t_p_text_orig


f = codecs.open("pg3300.txt","r","utf-8")

file_p_t, file_p_t_orig = tokenize_and_clean_text(paragrahp_file(f))

file_p_t_s = list(file_p_t)


def stemming(file):
    stemmer = PorterStemmer()
    for i_index,paragraph in enumerate(file):
        for j_index,word in enumerate(paragraph):
            paragraph[i_index][j_index] = stemmer.stem(word)

    return file

import gensim 