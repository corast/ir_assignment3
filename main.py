import InfRet as ir
import codecs
import gensim
import string
file = codecs.open("pg3300.txt","r","utf-8")

"""  
    ##########################################
    # Part 1, Data loading and preprocessing #
    ##########################################
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
matrix_sim_tfidf = gensim.similarities.MatrixSimilarity(tfidf_corpus)

#construct LSI model, number of topics 100. Paragraphs represented with a list of 100 pairs.
lsi_model = gensim.models.LsiModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=100)

lsi_corpus = lsi_model[tfidf_corpus]

matrix_sim_lsi = gensim.similarities.MatrixSimilarity(lsi_corpus) 
#print(lsi_model.show_topic(topicno=3,topn=3))
"""print(lsi_model.show_topic(0))"""
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

#Stemming query.
query_p_s = ir.stemming_q(query_p)

query_bow = dictionary.doc2bow(query_p_s)
#print(query_bow)

#convert BOW to TD-IDF representation.
#tfidf_model_q = gensim.models.TfidfModel(query_bow)
query_tfidf  = tfidf_model[query_bow]

#print(query_tfidf)

#print(dictionary.token2id)

#doc2similarity = enumerate(tfidf_index[query_tfidf])) 

#Use our matrix to search with the query on our documents.
sim = matrix_sim_tfidf[query_tfidf]
#Sort the result and return the top 3 results.
sims = ( sorted(enumerate(sim),key=lambda item: -item[1])[:3] )
print(sims)

def print_result(sims):
    for sim in sims:
        print("[Paragraph {0}]".format(sim[0]))
        index = sim[0]
        #Print 5 lines
        document = file_p_t_orig[index].split("\n")
        for i,line in enumerate(document):
            #if i reaches 5, we break the loop
            if(i == 5 or line == ""):
                break
            print(line)
        print()
            
print_result(sims)


#convert query tf_idf representation into lsi model

#vector representation of 
query_lsi = lsi_model[query_tfidf] 
#Print top 3 results.
print( sorted(query_lsi, key=lambda item: -abs(item[1]))[:3] )
#Print all result documents.
#print( lsi_model.show_topics() )

sim_lsi = matrix_sim_lsi[query_lsi]
sim_slsi = sorted(enumerate(sim_lsi),key=lambda item: -item[1])[:3]

def print_result2(sims):
    print("Simulated result lsi")
    for sim in sims:
        print("[topic {}]".format(sim[0]))
        
        topics = lsi_model.show_topic(sim[0])
        print("{0}*\"{1}\" + ".format(topics[1], topics[0]), end="")

        #print(topics)
        """for topic in topics[:-1]:
            print("{0}*\"{1}\" + ".format(topic[1], topic[0]), end="")
        print("{0}*\"{1}\"".format(topics[-1][1], topics[-1][0]))"""
        
print_result2(sim_slsi)