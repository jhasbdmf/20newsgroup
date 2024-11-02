import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import words
from textblob import TextBlob, Word
from english_words import get_english_words_set
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import string
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

'''#a preliminary data cleansing. 
    newsgroups_data_modified = newsgroups_data 
    for i in range(len(newsgroups_data_modified.data)):
        if "\n" in newsgroups_data_modified.data[i]:
            newsgroups_data_modified.data[i] = newsgroups_data_modified.data[i].replace("\n", " ")
        if "\t" in newsgroups_data_modified.data[i]:
            newsgroups_data_modified.data[i] = newsgroups_data_modified.data[i].replace("\t", ",")
        if "\\" in newsgroups_data_modified.data[i]:
            newsgroups_data_modified.data[i] = newsgroups_data_modified.data[i].replace("\\", " ")
        if "  " in newsgroups_data_modified.data[i]:
            newsgroups_data_modified.data[i] = newsgroups_data_modified.data[i].replace("  ", " ")
        print (type(newsgroups_data_modified)) 
'''


def generate_stop_words_list_from(extracted_words_frequency_distribution):

    extracted_words = extracted_words_frequency_distribution.keys()

    list_of_stop_words = ["a", "an", "the", "and", "it", "for", "or", "but", "in", "my", "your", "our", "their"]

    set_of_english_words = get_english_words_set(['web2','gcide'],lower=True)
    word_list = words.words()
    set_of_english_words = set_of_english_words.union(set(word_list))

    
    ps = PorterStemmer()
    set_of_stemmed_english_words = set()
    for i in set_of_english_words:
        set_of_stemmed_english_words.add(ps.stem(i))

    wordNet_lemmatizer = WordNetLemmatizer()
    part_of_speech_tags = ['a', 'r', 'n', 'v']


    for i in extracted_words:
        if extracted_words_frequency_distribution[i] > 1500:
            list_of_stop_words.append(i)
        
        elif not ps.stem(i) in set_of_english_words and not ps.stem(i) in set_of_stemmed_english_words:
            counter = 0
            for pos_tag in part_of_speech_tags:
                if not wordNet_lemmatizer.lemmatize(i, pos_tag) in set_of_english_words:
                    counter += 1 
            if counter == len(part_of_speech_tags):
                list_of_stop_words.append(i)

    return list_of_stop_words


def get_frequency_distribution_of_terms_from_given_a_dt_frequency_matrix (words, dtf_array):

    fdist = FreqDist()
    print (type(fdist))
    print ("first element is", dtf_array[0][0])

    for j in range(dtf_array.shape[1]):
        for i in range(dtf_array.shape[0]):
            fdist[words[j]] +=  dtf_array[i][j]
    '''
    for i in range(dtf_array.shape[0]):
        for j in range(dtf_array.shape[1]):
            fdist[words[i]] +=  dtf_array[i][j]
    '''
    print ("The term frequency distribution is computed \n")
    return fdist

def get_frequency_distribution_of_terms_from_given_a_dt_frequency_dataframe (df):
    
    sums_of_columns = df.sum(axis=0)
    fdist = FreqDist()
    words = list(df)

    for i in range(len(sums_of_columns)):
        fdist[words[i]] +=  sums_of_columns[i]

    return fdist


if __name__=="__main__":

    
    #part1
    newsgroups_data = fetch_20newsgroups()
    
    vectorizer = CountVectorizer(min_df=2)
    vectors = vectorizer.fit_transform(newsgroups_data.data)
    extracted_words = vectorizer.get_feature_names_out()
    print ('The shape of the term frequency matrix over the 20newsgroup collection is', vectors.shape)
    print('Dimensions of the aforementioned matrix are', extracted_words, "\n")
    
   

    #part2
    documents_terms_array = vectors.toarray()
    df = pd.DataFrame(data=documents_terms_array, columns=extracted_words)
    tf_distribution = get_frequency_distribution_of_terms_from_given_a_dt_frequency_dataframe(df)
    print("\nMost frequent words in the initial word list: \n", *tf_distribution.most_common(10), sep = "\n")



    print (len(extracted_words))
    #list_of_stop_words = generate_stop_words_list_from(extracted_words)
    list_of_stop_words = generate_stop_words_list_from (tf_distribution)
    print('The min size of the list of stop words the vectorizer is supposed to take into account is', len(list_of_stop_words))
   
    vectorizer_modified = CountVectorizer(min_df=2, stop_words=list_of_stop_words)
    vectors_modified = vectorizer_modified.fit_transform(newsgroups_data.data)

    documents_terms_array_modified = vectors_modified.toarray()
    print('The size of the list of all stop words the CountVectorizer actually took into account is', 
          len(vectorizer_modified.get_stop_words()), "+", len(vectorizer_modified.stop_words))
    
    print ('The shape of the tf matrix over 20newsgroup using stop_words', vectors_modified.shape)
   


    extracted_words_modified = vectorizer_modified.get_feature_names_out()
    df_modified = pd.DataFrame(data=documents_terms_array_modified, columns=extracted_words_modified)
    tf_distribution_modified = get_frequency_distribution_of_terms_from_given_a_dt_frequency_dataframe(df_modified)
    print("\n most frequent words in a modified list: \n", *tf_distribution_modified.most_common(10), sep = "\n")
    print("\n least frequent words in a modified list: \n", *tf_distribution_modified.most_common()[-10:], sep = "\n") 

    tf_distribution_modified.plot()

    
    
    #part3
    #text_categories = newsgroups_data.target_names

    
    
    text_categories = ['comp.graphics', 'comp.sys.ibm.pc.hardware', 
                        'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
                        'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.med', 
                        'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast']
    

    training_data = fetch_20newsgroups(subset="train", categories=text_categories)
    test_data = fetch_20newsgroups(subset="test", categories=text_categories)


    print("We have {} unique classes".format(len(text_categories)))
    print("We have the following unique classes: {}".format(text_categories))
    print("We have {} training samples".format(len(training_data.data)))
    print("We have {} test samples".format(len(test_data.data)))
 

    # Build the model
    classification_model = make_pipeline(vectorizer_modified, MultinomialNB())
    # Train the model using the training data
    classification_model.fit(training_data.data, training_data.target)
    # Predict the categories of the test data
    predicted_categories = classification_model.predict(test_data.data)
   
    
    print(len(np.array(test_data.target_names)[predicted_categories]))

    print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))
    precision = precision_score(test_data.target, predicted_categories, average='macro')
    print("The precision is {}".format(precision))
    recall = recall_score(test_data.target, predicted_categories, average='macro')
    print("The recall is {}".format(recall))
    
    f1 = f1_score(test_data.target, predicted_categories, average='macro')
    print("The F1-score is {}".format(f1))

    mat = confusion_matrix(test_data.target, predicted_categories)
    sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=training_data.target_names,yticklabels=training_data.target_names)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()
    
