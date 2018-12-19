import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as modsel
import matplotlib.pyplot as plt
import csv
import copy
import random
from math import *
param_grid_ = {'C': [0.00001, 0.0001, 0.001, 0.01, .1]}



class Classifier:

    def read(self, infile):
        self.docs = []
        self.readers = {}
        with open(infile) as csv_file:
            self.csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in self.csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.docs.append(row[11])
                    if row[3] not in self.readers:
                        self.readers[row[3]] = 1
                    else:
                        self.readers[row[3]] += 1


    def get_sample(self):
        readers = copy.deepcopy(self.readers)
        max_value = max(readers.values())
        for k, v in self.readers.items():
            if v == max_value:
                max_reader = k   # getting the reader with the most articles read
        self.best_reader = max_reader

    def get_sample_docs(self, infile, N):
        random.seed(N)
        with open(infile) as csv_file:
            self.csv_reader = csv.reader(csv_file, delimiter=',')
            sample_texts_read = []
            sample_texts_unread = []
            counter = 0
            counter2 = 0
            for row in self.csv_reader:
                if row[11] not in sample_texts_unread and row[11] not in sample_texts_read:
                    if row[3] == self.best_reader:
                        if counter2 < 200:
                            sample_texts_read.append([row[11], 0])
                            counter2 += 1
                    elif counter < 200:
                        sample_texts_unread.append([row[11], 1])
                        counter += 1
            self.sample_texts = sample_texts_read + sample_texts_unread
            random.shuffle(self.sample_texts)




    def get_data(self):

        self.training_data = self.sample_texts[0:320]
        self.training_texts = [x[0] for x in self.training_data]
        self.testing_data = self.sample_texts[320:400]
        self.testing_texts = [x[0] for x in self.testing_data]
        self.training_classes = [x[1] for x in self.training_data]
        self.testing_classes = [x[1] for x in self.testing_data]
        self.read_indexes = []
        self.unread_indexes = []
        for x in range(len(self.training_data)):
            if self.training_data[x][1] == 0:
                self.read_indexes.append(x)
            elif self.training_data[x][1] == 1:
                self.unread_indexes.append(x)

        self.class_prior = [len(self.read_indexes)/len(self.training_data), len(self.unread_indexes)/len(self.training_data)]



    """ Bag of Words """
    def generate_dicts_bag(self, MAX):

        self.tfCount = CountVectorizer(analyzer='word', max_features=MAX, stop_words='english')
        self.tfCount_matrix = self.tfCount.fit_transform(self.training_texts)

        self.tfCount_matrix_read = []
        self.tfCount_matrix_unread = []

        for i in self.read_indexes:
            self.tfCount_matrix_read.append(self.tfCount_matrix[i])

        for i in self.unread_indexes:
            self.tfCount_matrix_unread.append(self.tfCount_matrix[i])


        self.counts_read = np.sum(self.tfCount_matrix_read, axis=0).toarray().tolist()[0]
        self.counts_unread = np.sum(self.tfCount_matrix_unread, axis=0).toarray().tolist()[0]
        self.counts = [self.counts_read, self.counts_unread]
        self.word_counts = np.sum(self.tfCount_matrix, axis=0).tolist()[0]


        self.total_count = np.sum(self.tfCount_matrix)
        self.word_priors = [x / self.total_count for x in self.word_counts]

        self.dict = {}
        counter = 0
        for i in self.tfCount.get_feature_names():
            self.dict[i] = counter
            counter += 1

        self.F_bag = [[0] * len(self.dict) for _ in range(2)]

        self.tfCountT = CountVectorizer(analyzer='word', max_features=MAX, stop_words='english')
        self.tfCount_matrixT = self.tfCountT.fit_transform(self.testing_texts)



    def bag_of_words(self, alpha=1):

        for i in range(len(self.counts)):
            for word in range(len(self.dict)):
                prob = float(alpha + self.counts[i][word]) / float(sum(self.counts[i]) + (alpha * len(self.counts[i])))
                if prob == 0:
                    self.F_bag[i][word] = (0, log(self.word_priors[word]))
                else:
                    self.F_bag[i][word] = (log(prob), log(self.word_priors[word]))

        f_bag_read = np.save("bag_of_words_training_read", np.asarray(self.F_bag[0]))
        f_bag_unread = np.save("bag_of_words_training_unread", np.asarray(self.F_bag[1]))


    """ TFIDF """
    def generate_dicts_tfidf(self, MAX):

        self.tf = TfidfVectorizer(analyzer='word', max_features=MAX, stop_words='english', norm=None)
        self.tf_matrix = self.tf.fit_transform(self.training_texts)

        self.tf_matrix_read = []
        self.tf_matrix_unread = []

        for i in self.read_indexes:
            self.tf_matrix_read.append(self.tf_matrix[i])

        for i in self.unread_indexes:
            self.tf_matrix_unread.append(self.tf_matrix[i])

        self.tfidf_read = np.sum(self.tf_matrix_read, axis=0).toarray().tolist()[0]
        self.tfidf_unread = np.sum(self.tf_matrix_unread, axis=0).toarray().tolist()[0]

        self.tfidfs = [self.tfidf_read, self.tfidf_unread]

        self.tfidf_sums = np.sum(self.tf_matrix, axis=0).tolist()[0]
        self.tfidf_total = np.sum(self.tf_matrix)
        self.tfidf_priors = [x / self.tfidf_total for x in self.tfidf_sums]

        self.dict_tf = {}
        counter = 0
        for i in self.tf.get_feature_names():
            self.dict_tf[i] = counter
            counter += 1

        self.F_tf = [[0] * len(self.dict_tf) for _ in range(2)]

        self.tfT = TfidfVectorizer(analyzer='word', max_features=MAX, stop_words='english', norm=None)
        self.tf_matrixT = self.tfT.fit_transform(self.testing_texts)




    def tfidf(self, alpha=1):

        for i in range(len(self.tfidfs)):
            for word in range(len(self.dict_tf)):

                prob = float(alpha + (self.tfidfs[i][word])) / float(sum(self.tfidfs[i]) + (alpha * len(self.tfidfs[i])))
                if prob == 0:
                    self.F_tf[i][word] = (0, log(self.tfidf_priors[word]))
                else:
                    self.F_tf[i][word] = (log(prob), log(self.tfidf_priors[word]))

        f_tfidf_read = np.save("tfidf_training_read", np.asarray(self.F_tf[0]))
        f_tfidf_unread = np.save("tfidf_training_unread", np.asarray(self.F_tf[1]))


    def test(self):
        with open("bag_of_words_training_read.npy", "rb") as fread:
            read = np.load(fread)
        with open("bag_of_words_training_unread.npy", "rb") as funread:
            unread = np.load(funread)
        test_words = self.tfCountT.get_feature_names()
        predictions = [0] * len(self.testing_data)
        counter = 0
        for count_list in self.tfCount_matrixT.toarray():
            #get words in a given article
            words = []
            for word_index in range(len(count_list)):
                if count_list[word_index] > 0:
                    words.append(test_words[word_index])
            #intialize guess
            likeliest_class = [2, -float(inf)]
            read_likelihood = log(self.class_prior[0])
            unread_likelihood = log(self.class_prior[1])
            for word in words:
                if word in self.dict:
                    read_likelihood = read_likelihood + (float(read[self.dict[word]][0])) - (float(read[self.dict[word]][1]))
                    unread_likelihood = unread_likelihood + (float(unread[self.dict[word]][0])) - (float(unread[self.dict[word]][1]))
            if read_likelihood > unread_likelihood:
                likeliest_class = (0, read_likelihood)
            else:
                likeliest_class = (1, unread_likelihood)
            predictions[counter] = likeliest_class[0]
            counter = counter + 1
        correct_predictions = 0.0
        for i in range(len(self.testing_data)):
            if self.testing_data[i][1] == 0:
                if predictions[i] == 0:
                    correct_predictions = correct_predictions + 1
            elif self.testing_data[i][1] == 1:
                if predictions[i] == 1:
                    correct_predictions = correct_predictions + 1
            if predictions[i] == 2:
                print("error, guess never made")
        accuracy = float(correct_predictions/len(predictions))
        print (predictions, accuracy)
        return accuracy

    def tfidf_test(self):
        with open("tfidf_training_read.npy", 'rb') as fread:
            read = np.load(fread)
        with open("tfidf_training_unread.npy", 'rb') as funread:
            unread = np.load(funread)
        test_words = self.tfT.get_feature_names()
        predictions = [0] * len(self.testing_data)
        counter = 0


        for count_list in self.tf_matrixT.toarray():
            #get words in a given article
            words = []
            for word_index in range(len(count_list)):
                if count_list[word_index] > 0:
                    words.append(test_words[word_index])
            #initialize guess
            likeliest_rating = [0, -float(inf)]
            read_likelihood = log(self.class_prior[0])
            unread_likelihood = log(self.class_prior[1])
            for word in words:
                if word in self.dict_tf:
                    read_likelihood = read_likelihood + (float(read[self.dict_tf[word]][0])) - (float(read[self.dict_tf[word]][1]))
                    unread_likelihood = unread_likelihood + (float(unread[self.dict_tf[word]][0])) - (float(unread[self.dict_tf[word]][1]))
            if read_likelihood > unread_likelihood:
                likeliest_class = (0, read_likelihood)
            else:
                likeliest_class = (1, unread_likelihood)
            predictions[counter] = likeliest_class[0]
            counter = counter + 1
        correct_predictions = 0.0
        for i in range(len(self.testing_data)):
            if self.testing_data[i][1] == 0:
                if predictions[i] == 0:
                    correct_predictions = correct_predictions + 1
            elif self.testing_data[i][1] == 1:
                if predictions[i] == 1:
                    correct_predictions = correct_predictions + 1
            if predictions[i] == 2:
                print("error, guess never made")
        accuracy = float(correct_predictions/len(predictions))
        print (predictions, accuracy)
        return accuracy

    def plot_NB(self, trials):
        d = {'Bag of Words': np.zeros(trials),
             'TF-IDF': np.zeros(trials)}
        for i in range(trials):

            self.get_sample_docs('./shared_articles.csv', i)
            self.get_data()
            self.generate_dicts_bag(None)
            self.generate_dicts_tfidf(None)
            self.bag_of_words()
            self.tfidf()
            d['Bag of Words'][i] = self.test()
            d['TF-IDF'][i] = self.tfidf_test()

        df = pd.DataFrame.from_dict(d)
        sns.set_style("whitegrid")
        ax = sns.barplot(data=df, ci="sd")
        ax.set_ylabel("Accuracy", size=14)
        ax.tick_params(labelsize=14)
        ax.set_title("Bag of Words VS. TF-IDF", size=20)
        plt.show()



    def simple_logistic_classify(self, X_tr, y_tr, X_test, y_test, description, _C=1.0):
        # Helper function to train a logistic classifier and score on test data
        m = LogisticRegression(C=_C).fit(X_tr, y_tr)
        s = m.score(X_test, y_test)
        print ('Test score with', description, 'features:', s)
        return s

    def bow_logistic(self):

        x_te = self.tfCount.transform(self.testing_texts)
        self.simple_logistic_classify(self.tfCount_matrix, self.training_classes, x_te, self.testing_classes, 'bow')


    def tfidf_logistic(self):
        x_te = self.tf.transform(self.testing_texts)
        self.simple_logistic_classify(self.tf_matrix, self.training_classes, x_te, self.testing_classes, 'tfidf')


    def tune(self):
        self.bow_search = modsel.GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
        self.bow_search.fit(self.tfCount_matrix, self.training_classes)
        self.tfidf_search = modsel.GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
        self.tfidf_search.fit(self.tf_matrix, self.training_classes)
        self.search_results = pd.DataFrame.from_dict({
            'bow': self.bow_search.cv_results_['mean_test_score'],
            'tfidf': self.tfidf_search.cv_results_['mean_test_score']})

        
        print(self.search_results)

    def plot_tuned_logisitc(self):
        sns.set_style("whitegrid")
        ax = sns.boxplot(data=self.search_results, width=0.4)
        ax.set_ylabel('Accuracy', size=14)
        ax.tick_params(labelsize=14)
        plt.show()

    def test_tuned_logistic(self):
        x_te_bow = self.tfCount.transform(self.testing_texts)
        accuracy_bow = self.simple_logistic_classify(self.tfCount_matrix, self.training_classes, x_te_bow, self.testing_classes, 'bow',  _C=self.bow_search.best_params_['C'])
        x_te_tfidf = self.tf.transform(self.testing_texts)
        accuracy_tfidf = self.simple_logistic_classify(self.tf_matrix, self.training_classes, x_te_tfidf, self.testing_classes, 'tfidf', _C=self.tfidf_search.best_params_['C'])
        return (accuracy_bow, accuracy_tfidf)

    def collect_max_feature_data(self, trials, max_expo):
        data = {"bow_NB": np.zeros((trials, max_expo)),
                "tfidf_NB": np.zeros((trials, max_expo)),
                "bow_log": np.zeros((trials, max_expo)),
                "tfidf_log": np.zeros((trials, max_expo))}
        for trial in range(trials):
            for i in range(max_expo):
                max = 2**i
                self.get_sample_docs('./shared_articles.csv', trial)
                self.get_data()
                self.generate_dicts_bag(max)
                self.generate_dicts_tfidf(max)
                self.bag_of_words()
                self.tfidf()
                data["bow_NB"][trial, i] = self.test()
                data["tfidf_NB"][trial, i] = self.tfidf_test()
                self.tune()
                log_results = self.test_tuned_logistic()
                data["bow_log"][trial, i] = log_results[0]
                data["tfidf_log"][trial, i] = log_results[1]

        data_raw = np.save("raw_data", data)

        new_data = {"bow_NB": np.zeros(max_expo),
                "tfidf_NB": np.zeros(max_expo),
                "bow_log": np.zeros(max_expo),
                "tfidf_log": np.zeros(max_expo)}

        for index in data:
            new_data[index] = np.mean(data[index], axis=0)

        data_file = np.save("data_file", new_data)

        df = pd.DataFrame.from_dict(new_data)
        sns.set_style("whitegrid")
        ax = sns.lineplot(data=df)
        ax.set_ylabel("Accuracy", size=14)
        ax.set_xlabel("Maximum Features", size=14)
        ax.tick_params(labelsize=14)
        plt.show()






if __name__ == '__main__':
    c = Classifier()
    c.read('./shared_articles.csv')
    c.get_sample()
    c.get_sample_docs('./shared_articles.csv', 5)
    c.get_data()
    c.generate_dicts_bag(100000000)
    c.generate_dicts_tfidf(100000000)
    c.bag_of_words()
    c.test()
    c.tfidf()
    c.tfidf_test()
    c.plot_NB()
    c.bow_logistic()
    c.tfidf_logistic()
    c.tune()
    c.plot_tuned_logisitc()
    c.test_tuned_logistic()
    # c.collect_max_feature_data(5, 16)
