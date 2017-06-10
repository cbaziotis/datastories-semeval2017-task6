from __future__ import print_function

import csv
import itertools
import os
from collections import defaultdict
from random import random

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class SupervisedExpRunner(object):
    def __init__(self):
        self.model = None

    def _create_classifier(self):
        self.model = SVC(verbose=True)

    def _fit(self, X, y):
        if self.model is None:
            self._create_classifier()

        self.model.fit(X, y)

    def _evaluate(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        self.results = {'accuracy': acc}

    def _separate_data(self, X, y):
        X_win = X[y == 2]
        X_top10 = X[y == 1]
        X_rest = X[(y != 1) & (y != 2)]

        return X_win, X_top10, X_rest

    def _create_pairwise_data(self, Xs, ys, in_dv):
        X_pairs = []
        y_pairs = []
        for X, y in zip(Xs, ys):
            X_win, X_top10, X_rest = self._separate_data(in_dv.transform(X), y)

            for tweet_pair in itertools.product(X_win, X_top10):
                if random() > 0.5:
                    tweet_data = np.hstack((tweet_pair[0], tweet_pair[1]))
                    tweet_label = 1
                else:
                    tweet_data = np.hstack((tweet_pair[1], tweet_pair[0]))
                    tweet_label = 0

                X_pairs.append(tweet_data)
                y_pairs.append(tweet_label)

            for tweet_pair in itertools.product(X_top10, X_rest):
                if random() > 0.5:
                    tweet_data = np.hstack((tweet_pair[0], tweet_pair[1]))
                    tweet_label = 1
                else:
                    tweet_data = np.hstack((tweet_pair[1], tweet_pair[0]))
                    tweet_label = 0

                X_pairs.append(tweet_data)
                y_pairs.append(tweet_label)

            for tweet_pair in itertools.product(X_win, X_rest):
                if random() > 0.5:
                    tweet_data = np.hstack((tweet_pair[0], tweet_pair[1]))
                    tweet_label = 1
                else:
                    tweet_data = np.hstack((tweet_pair[1], tweet_pair[0]))
                    tweet_label = 0

                X_pairs.append(tweet_data)
                y_pairs.append(tweet_label)

        X = np.vstack(X_pairs)
        y = np.array(y_pairs)

        return X, y

    def get_results(self):
        return self.results

    def run_loo_exp(self, Xs, ys, ht_list, ow_name='results'):
        out_file = open(ow_name + '.csv', 'w')
        ow = csv.writer(out_file)
        micro_sum = 0
        total_pairs = 0
        num_hts = len(ys)
        for i in range(num_hts):
            print(str(100 * i / num_hts) + '% done')
            Xs_test = [Xs[i]]
            ys_test = [ys[i]]
            Xs_train = Xs[:i] + Xs[i + 1:]
            ys_train = ys[:i] + ys[i + 1:]

            dv = DictVectorizer(sparse=False)
            dv.fit([tweet for hashtag in Xs_train for tweet in hashtag])

            X_train, y_train = self._create_pairwise_data(Xs_train, ys_train, dv)
            X_test, y_test = self._create_pairwise_data(Xs_test, ys_test, dv)
            self._fit(X_train, y_train)
            self._evaluate(X_test, y_test)
            ht_result = self.get_results()
            ow.writerow([ht_list[i], str(ht_result['accuracy'])])
            print(ht_result)
            micro_sum += ht_result['accuracy'] * y_test.shape[0]
            total_pairs += y_test.shape[0]
            self.model = None
        out_file.close()
        print('100% done')
        return micro_sum / total_pairs


#########################
### Functions ###########
#########################
def create_bow_rep(in_tweet):
    bow_map = defaultdict(int)
    tokens = in_tweet.split()
    for tok in tokens:
        bow_map[tok] += 1
    return bow_map


def load_hashtag(data_location, htf):
    tweets = []
    labels = []
    for line in open(os.path.join(data_location, htf)).readlines():
        line_split = line.strip().split('\t')
        tweets.append(line_split[1])
        labels.append(int(line_split[2]))

    Y = np.array(labels)
    X_bow = [create_bow_rep(tweet) for tweet in tweets]

    return {'X_bow': X_bow, 'Y': Y}


def create_data(data_location):
    ht_files = sorted(os.listdir(data_location))

    Xs = []
    ys = []
    ht_list = []
    for htf in ht_files:
        ht_dict = load_hashtag(data_location, htf)

        ht_list.append(htf)
        ys.append(ht_dict['Y'])
        Xs.append(ht_dict['X_bow'])

    return Xs, ys, ht_list


#########################
### Main ################
#########################
def main():
    # if len(sys.argv) != 2:
    #     print('Usage:', __file__, '<data_dir>')
    #     print(' ', 'data_dir:', '\t', 'the path to the directory that contains the hahstag data files')
    #     sys.exit(1)
    #
    # data_location = sys.argv[1]
    data_location = "trial_data"
    Xs, ys, ht_list = create_data(data_location)

    exp_runner = SupervisedExpRunner()
    outwriter_name = 'results_sem'
    results = exp_runner.run_loo_exp(Xs, ys, ht_list, outwriter_name)
    print('micro-avergae accuracy:', results)


if __name__ == '__main__':
    main()
