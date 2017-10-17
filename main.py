import InfRet as ir
import codecs
import gensim
import string
file = codecs.open("pg3300.txt","r","utf-8")

"""  
    #########################################
    # art 1, Data loading and preprocessing #
    #########################################
"""
file_p_t, file_p_t_orig = ir.tokenize_and_clean_text(ir.paragrahp_file(file))

#file_p_t is the file where each paragraph is in list form.
#file_p_t_orig is the original file, split into lists of pargraphs from the file.
#for x in range(0,5):
    #print("f:",file_p_t[x])

file_p_t_s = ir.stemming(file_p_t)
#print(file_p_t_s)

#for x in range(0,5):
    #print("e:",file_p_t_s[x])

"""  
    ###############################
    # Part 2, Dictionary building #
    ###############################
"""

dictionary = ir.build_dictionary(file_p_t_s)

stopword_list = ir.create_stopword_list()

ir.remove_stopwords(dictionary, stopword_list)

#corpus = [dictionary.doc2bow(paragaph) for paragaph in file_p_t_s]

corpus = []
for paragraph in file_p_t_s:
    """Create 'bag of words' """
    corpus.append(dictionary.doc2bow(paragraph))

"""  
    ############################
    # Part 3, Retrieval Models #
    ############################
"""

#convert bag-of-words into TF-IDF weights then LSI(Latent Semantic Indexing)
#weigths

#Build TF-IDF model from copus.
tfidf_model = gensim.models.TfidfModel(corpus)


# Map Bags-of-Words into TF-IDF weights (now each paragraph should be represented with a list of
#pairs (word-index, word-weight) ). Some useful code:
tfidf_corpus = tfidf_model[corpus]

#Construct MatrixSimilarity object to calculate similarities
#between paragraphs and queries.
matrix_sim = gensim.similarities.MatrixSimilarity(tfidf_corpus)

#construct LSI model, number of topics 100. Paragraphs represented with a list of 100 pairs.
lsi_model = gensim.models.LsiModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=100)

lsi_corpus = lsi_model[tfidf_corpus]

m_sim = gensim.similarities.MatrixSimilarity(lsi_corpus) 
#print(lsi_model.show_topic(topicno=3,topn=3))
print(lsi_model.show_topic(0))
#[('0', 0.72653483992568624), ('1', 0.25767692702039041), ('2', 0.2425337297775341)]
#Output are the top 3 words associated with 3 random topics. 
#The decimal number is the weight of the multiplied number.
#i.e how much this word influence the particular topic.
#print(lsi_model.show_topic(0))
"""  
    ####################
    # Part 4. Querying #
    ####################
"""
#query = "What is the function of money?"
query = "How taxes influence Economics?"
query_p = ir.preprocessing(query)
print(query_p)

#create bag of words from the query using dictionary
#query_b = dictionary.doc2bow(['how','taxes','influence','economics'])

query_bow = dictionary.doc2bow(query_p)
#print(query_bow)

#convert BOW to TD-IDF representation.
#tfidf_model_q = gensim.models.TfidfModel(query_bow)
tfidf_corpus_q = tfidf_model[query_bow]
print(tfidf_corpus_q)

#print(dictionary.token2id)
print(dictionary)