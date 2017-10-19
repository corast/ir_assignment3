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
    """Tokenize the text, clean and remove paragraphs containing (G/g)utenberg  """
    #Empty arrays for storage.
    t_p_text = []
    t_p_text_orig = []

    #Table of characters to remove from our string, since not every character is present from string.punction, 
    # we have to add some extra, like newline, rawtext, tabs and some special once like the pound sign(probably more hidden in the text)
    table = str.maketrans(dict.fromkeys(string.punctuation+"\n\r\t£"))

    for para in t_file:
        #Remove paragraphs that contains the word “Gutenberg” or “gutenberg”
        if para.__contains__('Gutenberg') or para.__contains__('gutenberg'):
            continue
        #turn paragraph into lowercase letters.
        para_lower = para.lower()

        #Since we are dealing with whole paragraphs at a time, removing \n from the paragraph, merges two words together
        #To handle this we just need to replace \n with a space ' '
        para_lower = para_lower.replace('\n',' ')

        #if the paragrahp contains '-' with words like a-hunter in paragraph[0], we need to specificly handle it
        if para_lower.__contains__('-'):
            para_lower = para_lower.replace("-"," ")


        #clean the text of paragraphs and \n\r\t characters
        para_clean = para_lower.translate(table)

        #Keep copy of the original paragrahp unaltered.
        t_p_text_orig.append(para)   

        #Add the list split into words to the list
        t_p_text.append(para_clean.split())

    return t_p_text, t_p_text_orig

def stemming(file):
    """ Return words to their origin word. """
    stemmer = PorterStemmer()
    for i_index, paragraph in enumerate(file):
        for j_index, word in enumerate(paragraph):
            file[i_index][j_index] = stemmer.stem(word)

    return file

def stemming_q(query):
    """ Return a query which is stemmed """
    stemmer = PorterStemmer()
    for i_index, word in enumerate(query):
        query[i_index] = stemmer.stem(word)
    return query


# Part 2, Dictionary building

def build_dictionary(document):
    """ Build dictionary from a document where each paragragh is a list of words. """
    dictionary = gensim.corpora.Dictionary(document)
    return dictionary



def create_stopword_list():
    """ read from the file and split at the character ',' returning list of stopwords """
    stopwords = codecs.open("common-english-words.txt","r","utf-8")
    s_file = stopwords.read()
    return s_file.split(',')


#Remove stopwords.
def remove_stopwords(dictionary, stoplist):
    """ remove stopwords """
    stop_ids = []
    for stopword in stoplist:
        if stopword in dictionary.token2id:
            stop_ids.append(dictionary.token2id[stopword])
    #filter out stopwords from the dictionary.
    dictionary.filter_tokens(stop_ids)
    return dictionary


#preprosess query
def preprocessing(query):
    """ Remove punctiation, tokenize, lowercase """
    #Table of characters to remove from our string
    table = str.maketrans(dict.fromkeys(string.punctuation+"\n\r\t"))

    #turn query into lowercase letters.
    para_lower = query.lower()

    #clean the text of the query and \n\r\t characters
    para_clean = para_lower.translate(table)

    #Return tokenized query.
    return para_clean.split()