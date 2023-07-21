from MR_functions import *

#changing the parameters below all the code references are changed and a plot for comparison is created at the end
voc_size = voc_stopwords_size = voc_stemming_size = voc_stopwords_stemmed_size = 1000

#name of the stored file to have a quicker view over them
vocname = 'Movie Reviews/code/Generated/vocabulary.txt'
vocname_stopwords = 'Movie Reviews/code/Generated/vocabulary_stopwords.txt'
vocname_stemming = 'Movie Reviews/code/Generated/vocabulary_stemming.txt'
vocname_stopwords_stemmed = 'Movie Reviews/code/Generated/vocabulary_stopwords_stemmed.txt'

#directories paths
pos_test_dir_bow = 'Movie Reviews/code/Dataset/test/pos/'
neg_test_dir_bow = 'Movie Reviews/code/Dataset/test/neg/'
pos_valid_dir_bow = 'Movie Reviews/code/Dataset/validation/pos/'
neg_valid_dir_bow = 'Movie Reviews/code/Dataset/validation/neg/'
pos_train_dir_bow = 'Movie Reviews/code/Dataset/train/pos/'
neg_train_dir_bow = 'Movie Reviews/code/Dataset/train/neg/'

############################### CLASSIC MODEL ###############################

#build the vocabulary
cnt = collections.Counter()
cnt = dir_words_count(neg_train_dir_bow)
cnt.update(dir_words_count(pos_train_dir_bow))
create_vocabulary(cnt, voc_size, vocname)

#bow for test, train and validation datasets
bow_test = bag_of_words_dir(pos_test_dir_bow, neg_test_dir_bow, vocname)
np.savetxt('Movie Reviews/code/Generated/bow_test.txt.gz', bow_test)
bow_validation = bag_of_words_dir(pos_valid_dir_bow, neg_valid_dir_bow, vocname)
np.savetxt('Movie Reviews/code/Generated/bow_validation.txt.gz', bow_validation)
bow_train = bag_of_words_dir(pos_train_dir_bow, neg_train_dir_bow, vocname)
np.savetxt('Movie Reviews/code/Generated/bow_train.txt.gz', bow_train)

############################# STOPWORDS MODEL #################################

#create a new counter ignoring very common words
stopwords = read_file('Movie Reviews/code/Generated/stopwords.txt')
cnt_stopwords = cnt.copy()

for w in stopwords:
    if w in cnt_stopwords:
        del cnt_stopwords[w]

#create new vocabulary without stopwords
create_vocabulary(cnt_stopwords, voc_stopwords_size, vocname_stopwords)

#create new bow
bow_train_stopwords = bag_of_words_dir(pos_train_dir_bow, neg_train_dir_bow, vocname_stopwords)
np.savetxt('Movie Reviews/code/Generated/bow_train_stopwords.txt.gz', bow_train_stopwords)
bow_test_stopwords = bag_of_words_dir(pos_test_dir_bow, neg_test_dir_bow, vocname_stopwords)
np.savetxt('Movie Reviews/code/Generated/bow_test_stopwords.txt.gz', bow_test_stopwords)

############################# STEMMED MODEL #################################

#create the new counter
cnt_stemmed = dir_words_count(neg_train_dir_bow, stemming=True)
cnt_stemmed.update(dir_words_count(pos_train_dir_bow, stemming=True))

#create new vocabulary after stemming
create_vocabulary(cnt_stemmed, voc_stemming_size, vocname_stemming)

#create new bow
bow_train_stemming = bag_of_words_dir(pos_train_dir_bow, neg_train_dir_bow, vocname_stemming, stemming = True)
np.savetxt('Movie Reviews/code/Generated/bow_train_stemming.txt.gz', bow_train_stemming)
bow_test_stemming = bag_of_words_dir(pos_test_dir_bow, neg_test_dir_bow, vocname_stemming, stemming = True)
np.savetxt('Movie Reviews/code/Generated/bow_test_stemming.txt.gz', bow_test_stemming)

############################# STEMMING + STOPWORDS #################################

#create the new counter
cnt_stopwords_stemmed = collections.Counter()
for w, c in cnt_stopwords.items():
    w = stem(w)
    cnt_stopwords_stemmed[w] += c   
    
#create new vocabulary after stemming
create_vocabulary(cnt_stopwords_stemmed, voc_stopwords_stemmed_size, vocname_stopwords_stemmed)

#create new bow
bow_train_stemming = bag_of_words_dir(pos_train_dir_bow, neg_train_dir_bow, vocname_stopwords_stemmed, stemming = True)
np.savetxt('Movie Reviews/code/Generated/bow_train_stopwords_stemmed.txt.gz', bow_train_stemming)
bow_test_stemming = bag_of_words_dir(pos_test_dir_bow, neg_test_dir_bow, vocname_stopwords_stemmed, stemming = True)
np.savetxt('Movie Reviews/code/Generated/bow_test_stopwords_stemmed.txt.gz', bow_test_stemming)


#check soundness of the counters
'''
print(cnt_stopwords_stemmed)
print(len(cnt_stopwords_stemmed))
print(sum(cnt_stopwords_stemmed.values()))
print(cnt_stemmed)
print(len(cnt_stemmed))
print(sum(cnt_stemmed.values()))
print(cnt_stopwords)
print(len(cnt_stopwords))
print(sum(cnt_stopwords.values()))
print(cnt)
print(len(cnt))
print(sum(cnt.values()))
'''