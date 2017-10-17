#Part 1, Data loading and preprocessing
#Testing
#Imports part 1
import random; random.seed(123)
import string
import codecs

from nltk.stem.porter import PorterStemmer
#Imports part2
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 

import time

#Partition file into separate paragraphs. Paragraphs are text chunks separated by empty line.
#Partitioning result should be a list of paragraphs. For us, each paragraph will be a separate document.

# p - paragraphed, t - tokenized and cleaned of characters, s - stemmed

def paragrahp_file(file):
    """ Pre process file and return all paragraphs in an list"""
    text = []
    paragraph = ""
    for line in file:
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

    #TODO: fiks ord som henger sammen med bildestrek, som egentlig skal være unike ord.

    #Table of characters to remove from our string
    table = str.maketrans(dict.fromkeys(string.punctuation+"\n\r\t"))

    for para in t_file:
        """ Remove (filter out) paragraphs containing the word “Gutenberg” (=headers and footers)"""
        if para.__contains__('Gutenberg') or para.__contains__('gutenberg'):
            continue
        #turn paragraph into lowercase letters.
        para_lower = para.lower()
        #clean the text of paragraphs and \n\r\t characters
        #print("cleaning:", para_lower)
        for word in para_lower:
            if word.__contains__('-'):
                print(para_lower)
                #print(word)
        
        para_clean = para_lower.translate(table)

        #print("cleaned:",para_clean)

        t_p_text_orig.append(para_clean)   

        #Add the list split into words to the list
        t_p_text.append(para_clean.split())

    return t_p_text, t_p_text_orig


"""f = codecs.open("pg3300.txt","r","utf-8")"""

#file_p_t_orig er originalen hvor vi ikke har klippet bort stemming
"""file_p_t, file_p_t_orig = tokenize_and_clean_text(paragrahp_file(f))"""

#file_p_t is the file where each paragraph is in list form.

def stemming(file):
    """ Return words to their origin word. """
    stemmer = PorterStemmer()
    for i_index, paragraph in enumerate(file):
        for j_index, word in enumerate(paragraph):
            file[i_index][j_index] = stemmer.stem(word)

    return file


"""file_p_t_s = stemming(file_p_t)"""

# Part 2, Dictionary building

def build_dictionary(document):
    """ Build dictionary from a document where each paragragh is a list of words. """
    dictionary = gensim.corpora.Dictionary(document)
    dictionary.save('testing.dict')
    return dictionary



#start_time = time.time()
#dictionary = gensim.corpora.Dictionary(file_p_t_s)
#elapse_time = time.time() - start_time
#print("Det tok:",format(elapse_time))

"""dictionary = build_dictionary(file_p_t_s)"""
#print(dictionary)

#print(stopwords.read())

def create_stopword_list():
    """ read from the buffer and split at the character ',' returning list of stopwords """
    stopwords = codecs.open("common-english-words.txt","r","utf-8")
    s_file = stopwords.read()
    return s_file.split(',')

"""stopword_list = create_stopword_list()"""



#Remove stopwords.
def remove_stopwords(dictionary, stoplist):
    stop_ids = []
    for stopword in stoplist:
        if stopword in dictionary.token2id:
            stop_ids.append(dictionary.token2id[stopword])
    #filter out stopwords from the dictionary.
    dictionary.filter_tokens(stop_ids)
    return dictionary

"""remove_stopwords(dictionary, stopword_list)"""

#preprosess query
def preprocessing(query):
    """ Remove punctiation, tokenize, stemming """
    #Table of characters to remove from our string
    table = str.maketrans(dict.fromkeys(string.punctuation+"\n\r\t"))

    #turn query into lowercase letters.
    para_lower = query.lower()

    #clean the text of the query and \n\r\t characters
    para_clean = para_lower.translate(table)

    #Return tokenized cleander query.
    return para_clean.split()