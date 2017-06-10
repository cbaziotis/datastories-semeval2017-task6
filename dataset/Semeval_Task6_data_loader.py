"""
Created by Christos Baziotis.
"""
import glob
import itertools
import os
from random import random

from utilities.generic import clean_text


class SemEval2017Task6:
    # Validation set is used for tuning the parameters of a model
    # Test set is used for performance evaluation

    # Training set      --> to fit the parameters [i.e., weights]
    # Validation set    --> to tune the parameters [i.e., architecture]
    # Test set          --> to assess the performance [i.e., generalization and predictive power]

    def __init__(self):
        pass

        self.SEPARATOR = "\t"
        self.directories = ["trial_data", "train_data"]
        print()

    @staticmethod
    def _separate_data(data):
        """
        dictionary in the form {tweet_id:(text,score)}
        :param data:
        :return: list of texts (only) divided by score
        """
        win = [(k, v[0]) for k, v in data.items() if v[1] == "2"]
        top10 = [(k, v[0]) for k, v in data.items() if v[1] == "1"]
        rest = [(k, v[0]) for k, v in data.items() if v[1] != "1" and v[1] != "2"]

        return win, top10, rest

    @staticmethod
    def pairwise_groups(a, b):
        X_pairs = []
        id_pairs = []
        y_pairs = []
        for tweet_pair in itertools.product(a, b):
            if random() > 0.5:
                tweet_data = (tweet_pair[0][1], tweet_pair[1][1])
                tweet_ids = (tweet_pair[0][0], tweet_pair[1][0])
                tweet_label = 1
            else:
                tweet_data = (tweet_pair[1][1], tweet_pair[0][1])
                tweet_ids = (tweet_pair[1][0], tweet_pair[0][0])
                tweet_label = 0

            X_pairs.append(tweet_data)
            id_pairs.append(tweet_ids)
            y_pairs.append(tweet_label)

        return X_pairs, id_pairs, y_pairs

    def _create_pairwise_data(self, data):
        """
        dictionary in the form {tweet_id:(text,score)}
        :param data:
        :return: 3 lists
        X_pairs: list of tuples (text1,text2)
        y_pairs: list of labels {0,1,2}
        id_pairs: list of tuples (id1,id2) with 1-1 relation to the texts of X_pairs
        """
        X_win, X_top10, X_rest = self._separate_data(data)

        X_pairs = []
        id_pairs = []
        y_pairs = []

        _x, _id, _y = self.pairwise_groups(X_win, X_top10)
        X_pairs += _x
        id_pairs += _id
        y_pairs += _y

        _x, _id, _y = self.pairwise_groups(X_top10, X_rest)
        X_pairs += _x
        id_pairs += _id
        y_pairs += _y

        _x, _id, _y = self.pairwise_groups(X_win, X_rest)
        X_pairs += _x
        id_pairs += _id
        y_pairs += _y

        return X_pairs, y_pairs, id_pairs

    def parse_file(self, filename, unlabeled=False):
        """
        Reads the text file and returns a dictionary in the form [(text,label)]
        :param filename: the complete file name
        :return:
        """
        data = {}
        fname_print_friendly = filename.split("/")[-1].split("\\")[-1]
        print("Parsing file:", fname_print_friendly, end=" ")
        for line_id, line in enumerate(open(filename, "r", encoding="utf-8").readlines()):

            try:
                columns = line.rstrip().split(self.SEPARATOR)
                tweet_id = columns[0]
                text = clean_text(columns[1])

                if unlabeled:
                    if text != "Not Available":
                        data[tweet_id] = text
                else:
                    sentiment = columns[2]
                    if text != "Not Available":
                        data[tweet_id] = (text, sentiment)
            except Exception as e:
                print("\nWrong format in line:{} in file:{}".format(line_id, fname_print_friendly))
                raise Exception

        print("done!")
        return data

    def get_test_data_task_1(self):
        wd = os.path.dirname(__file__)
        ht_files = [f for f in sorted(glob.iglob('{}/{}/**/*.tsv'.format(wd, "evaluation_dir"), recursive=True))]

        tweets = {}
        for htf in ht_files:
            fname = htf.split("/")[-1].split("\\")[-1]
            hashtag_data = self.parse_file(htf, unlabeled=True)

            data = [(k, v) for k, v in hashtag_data.items()]
            pairs = list(itertools.combinations(data, 2))

            tweets[fname] = [((p[0][0], p[1][0]), (p[0][1], p[1][1])) for p in pairs]

        return tweets

    def get_training_data_task_1(self):
        """
        Returns a dictionary in the form hashtag_filename: tuple of 3 lists (text_pair,label,id_pair)
            text_pair: list of tuples (text1,text2)
            label: list of int/str with values from {0,1} 1-> text1 is funnier, 0-> text2 is funnier
            id_pair: list of tuples (id_of_text1,id_of_text2)
        :return:
        """
        wd = os.path.dirname(__file__)
        ht_files = [f for d in self.directories
                    for f in sorted(glob.iglob('{}/{}/**/*.tsv'.format(wd, d), recursive=True))]

        tweets = {}
        twts = 0
        for htf in ht_files:
            fname = htf.split("/")[-1].split("\\")[-1]
            hashtag_data = self.parse_file(htf)
            twts += len(hashtag_data)
            tweets[fname] = self._create_pairwise_data(hashtag_data)

        # return tweets
        return {k: tweets[k] for k in list(tweets.keys())[:20]}

    def get_all_training_data_task_1(self):
        """
        Returns a dictionary in the form hashtag_filename: tuple of 3 lists (text_pair,label,id_pair)
            text_pair: list of tuples (text1,text2)
            label: list of int/str with values from {0,1} 1-> text1 is funnier, 0-> text2 is funnier
            id_pair: list of tuples (id_of_text1,id_of_text2)
        :return:
        """
        wd = os.path.dirname(__file__)
        ht_files = [f for d in ["trial_data", "train_data", "gold_labels"]
                    for f in sorted(glob.iglob('{}/{}/**/*.tsv'.format(wd, d), recursive=True))]

        tweets = {}
        for htf in ht_files:
            fname = htf.split("/")[-1].split("\\")[-1]
            hashtag_data = self.parse_file(htf)
            tweets[fname] = self._create_pairwise_data(hashtag_data)

        return tweets


    def get_gold_data_task_1(self):
        """
        Returns a dictionary in the form hashtag_filename: tuple of 3 lists (text_pair,label,id_pair)
            text_pair: list of tuples (text1,text2)
            label: list of int/str with values from {0,1} 1-> text1 is funnier, 0-> text2 is funnier
            id_pair: list of tuples (id_of_text1,id_of_text2)
        :return:
        """
        wd = os.path.dirname(__file__)
        ht_files = [f for f in sorted(glob.iglob('{}/{}/**/*.tsv'.format(wd, "gold_labels"), recursive=True))]

        tweets = {}
        for htf in ht_files:
            fname = htf.split("/")[-1].split("\\")[-1]
            hashtag_data = self.parse_file(htf)
            tweets[fname] = self._create_pairwise_data(hashtag_data)

        return tweets


    def get_training_data_task_2(self):
        """
        Returns a dictionary in the form hashtag_filename: tuple of 3 lists (texts,labels,ids)
        :return:
        """
        wd = os.path.dirname(__file__)
        ht_files = [f for d in self.directories
                    for f in sorted(glob.iglob('{}/{}/**/*.tsv'.format(wd, d), recursive=True))]

        tweets = {}
        for htf in ht_files:
            fname = htf.split("/")[-1].split("\\")[-1]
            hashtag_data = self.parse_file(htf)
            data = [(v[0], v[1], k) for k, v in hashtag_data.items()]
            texts = [d[0] for d in data]
            labels = [d[1] for d in data]
            ids = [d[2] for d in data]
            tweets[fname] = (texts, labels, ids)
        return tweets


# task1 = SemEval2017Task6().get_gold_data_task_1()
# task1 = SemEval2017Task6().get_test_data_task_1()
# task1 = SemEval2017Task6().get_training_data_task_1()
# task1 = SemEval2017Task6().get_all_training_data_task_1()
# task2 = SemEval2017Task6().get_training_data_task_2()
# print()
