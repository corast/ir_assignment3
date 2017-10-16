import InfRet as ir
import codecs

file = codecs.open("pg3300.txt","r","utf-8")
#Part 1, Data loading and preprocessing
file_p_t, file_p_t_orig = ir.tokenize_and_clean_text(ir.paragrahp_file(file))

#file_p_t is the file where each paragraph is in list form.
#file_p_t_orig is the original file, split into lists of pargraphs from the file.

file_p_t_s = ir.stemming(file_p_t)
# Part 2, Dictionary building
dictionary = ir.build_dictionary(file_p_t_s)

stopword_list = ir.create_stopword_list()

ir.remove_stopwords(dictionary, stopword_list)