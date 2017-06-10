"""
Created by Christos Baziotis.
"""
import random

from ekphrasis.classes.textpp import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from kutilities.helpers.data_preparation import print_dataset_statistics, labels_to_categories, categories_to_onehot

from dataset.Semeval_Task6_data_loader import SemEval2017Task6
from sk_transformers.CustomPreProcessor import CustomPreProcessor
from sk_transformers.EmbeddingsExtractor import EmbeddingsExtractor
from utilities.ignore_warnings import set_ignores

set_ignores()
from sklearn.pipeline import Pipeline
import numpy
from embeddings.WordVectorsManager import WordVectorsManager


def prepare_dataset(X, y, pipeline, y_one_hot=True, y_as_is=False):
    try:
        print_dataset_statistics(y)
    except:
        pass

    X = pipeline.fit_transform(X)

    if y_as_is:
        try:
            return X, numpy.asarray(y, dtype=float)
        except:
            return X, y

    # 1 - Labels to categories
    y_cat = labels_to_categories(y)

    if y_one_hot:
        # 2 - Labels to one-hot vectors
        return X, categories_to_onehot(y_cat)

    return X, y_cat


def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(corpus, dim).read()
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    wv_map = {}
    pos = 0
    # +1 for zero padding token and +1 for unk
    emb_matrix = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        pos = i + 1
        wv_map[word] = pos
        emb_matrix[pos] = vector

    # add unknown token
    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)

    return emb_matrix, wv_map


def prepare_text_only_dataset(X, pipeline):
    X = pipeline.fit_transform(X)
    return X


def data_splits(dataset, final=False):
    '''
    Splits a dataset in parts
    :param dataset:
    :param final: Flag that indicates if we want a split for tha final submission or for normal training
    :return:
    '''
    if final:
        # 95% training and 5% validation
        train_ratio = 0.95
        train_split_index = int(len(dataset) * train_ratio)

        training = dataset[:train_split_index]
        test = dataset[train_split_index:]

        return training, test
    else:
        # 80% training, 10% validation and 10% testing
        train_ratio = 0.8
        val_test_ratio = 0.5
        train_split_index = int(len(dataset) * train_ratio)
        val_test_split_index = int((len(dataset) - train_split_index) * val_test_ratio)

        training = dataset[:train_split_index]
        rest = dataset[train_split_index:]
        validation = rest[:val_test_split_index]
        test = rest[val_test_split_index:]

        return training, validation, test


class Task6Loader:
    """
    Task 6: #HashtagWars: Learning a Sense of Humor
    """

    def __init__(self, word_indices, text_lengths, subtask="1", **kwargs):

        self.word_indices = word_indices
        self.subtask = subtask
        self.text_lengths = text_lengths
        self.y_one_hot = kwargs.get("y_one_hot", True)
        self.own_vectors = kwargs.get("own_vectors", True)
        self.y_padding = kwargs.get("y_padding", None)  # for subtask2

        if self.own_vectors:
            add_tokens = (True, True) if subtask == "1" else True
        else:
            add_tokens = (False, False) if subtask == "1" else False

        # configure the text preprocessing pipeling using ekphrasis and scikit-learn
        self.pipeline = Pipeline([
            ('preprocess', CustomPreProcessor(TextPreProcessor(
                backoff=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
                include_tags={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
                fix_html=True,
                segmenter="twitter",
                corrector="twitter",
                unpack_hashtags=True,
                unpack_contractions=True,
                spell_correct_elong=False,
                tokenizer=SocialTokenizer(lowercase=True).tokenize,
                dicts=[emoticons]))),
            ('ext', EmbeddingsExtractor(word_indices=word_indices,
                                        max_lengths=text_lengths,
                                        hierarchical=subtask == "2",
                                        add_tokens=add_tokens,
                                        unk_policy="random"))])

        # loading data
        print("Loading data...")
        if subtask == "1":
            dataset = SemEval2017Task6().get_training_data_task_1()
        else:
            dataset = SemEval2017Task6().get_training_data_task_2()
        self.dataset = [v for k, v in sorted(dataset.items())]
        random.Random(42).shuffle(self.dataset)

        # [x for hashtag in dataset for x in hashtag[0]]
        try:
            print("total observations:", len([x for hashtag in dataset for x in hashtag[1]]))
        except:
            pass

    def pad_X(self, X):
        data = numpy.zeros((len(X), self.y_padding, self.text_lengths), dtype='int32')
        for i, hashtag in enumerate(X):
            for j, sentence in enumerate(hashtag):
                data[i, j] = sentence
        return data

    def load_train_val_test(self, only_test=False):
        """
        Load data for normal model training
        :return:
        """
        train, val, test = data_splits(self.dataset)
        if self.subtask == "1":
            X_train = [x for hashtag in train for x in hashtag[0]]
            y_train = [x for hashtag in train for x in hashtag[1]]

            X_val = [x for hashtag in val for x in hashtag[0]]
            y_val = [x for hashtag in val for x in hashtag[1]]

            X_test = [x for hashtag in test for x in hashtag[0]]
            y_test = [x for hashtag in test for x in hashtag[1]]
        else:
            X_train = [hashtag[0] for hashtag in train]
            y_train = [[int(h) for h in hashtag[1]] for hashtag in train]

            X_val = [hashtag[0] for hashtag in val]
            y_val = [[int(h) for h in hashtag[1]] for hashtag in val]

            X_test = [hashtag[0] for hashtag in test]
            y_test = [[int(h) for h in hashtag[1]] for hashtag in test]

            if self.y_padding:
                y_train = [numpy.pad(yt, (0, self.y_padding - len(yt) % self.y_padding), 'constant') for yt in y_train]
                y_val = [numpy.pad(yt, (0, self.y_padding - len(yt) % self.y_padding), 'constant') for yt in y_val]
                y_test = [numpy.pad(yt, (0, self.y_padding - len(yt) % self.y_padding), 'constant') for yt in y_test]

                y_train = [numpy.divide(yt, yt.max()) for yt in y_train]
                y_val = [numpy.divide(yt, yt.max()) for yt in y_val]
                y_test = [numpy.divide(yt, yt.max()) for yt in y_test]

        if not only_test:
            print("\nPreparing training set...")
            training = prepare_dataset(X_train, y_train, self.pipeline, self.y_one_hot)
            print("\nPreparing validation set...")
            validation = prepare_dataset(X_val, y_val, self.pipeline, self.y_one_hot)
        print("\nPreparing test set...")
        testing = prepare_dataset(X_test, y_test, self.pipeline, self.y_one_hot)

        if only_test:
            return testing
        else:
            return training, validation, testing

    def load_loo(self):
        """
        Load data for Leave-One-Out validation
        :return:
        """

        data = []
        hashtags = []
        for hashtag, values in SemEval2017Task6().get_all_training_data_task_1().items():
            prepared = prepare_dataset(values[0], values[1], self.pipeline, self.y_one_hot)
            data.append(prepared)
            hashtags.append(hashtag)
        data = numpy.asarray(data)
        hashtags = numpy.asarray(hashtags)

        return data, hashtags

    def load_final(self):
        """
        Load data for training for the final submission
        :return:
        """
        train, test = data_splits(self.dataset, final=True)

        if self.subtask == "1":
            X_train = [x for hashtag in train for x in hashtag[0]]
            y_train = [x for hashtag in train for x in hashtag[1]]

            X_test = [x for hashtag in test for x in hashtag[0]]
            y_test = [x for hashtag in test for x in hashtag[1]]
        else:
            X_train = [hashtag[0] for hashtag in train]
            y_train = [[int(h) for h in hashtag[1]] for hashtag in train]

            X_test = [hashtag[0] for hashtag in test]
            y_test = [[int(h) for h in hashtag[1]] for hashtag in test]

            if self.y_padding:
                y_train = [numpy.pad(yt, (0, self.y_padding - len(yt) % self.y_padding), 'constant') for yt in y_train]
                y_test = [numpy.pad(yt, (0, self.y_padding - len(yt) % self.y_padding), 'constant') for yt in y_test]

                y_train = [numpy.divide(yt, yt.max()) for yt in y_train]
                y_test = [numpy.divide(yt, yt.max()) for yt in y_test]

        print("\nPreparing training set...")
        training = prepare_dataset(X_train, y_train, self.pipeline, self.y_one_hot, y_as_is=self.subtask == "2")
        print("\nPreparing test set...")
        testing = prepare_dataset(X_test, y_test, self.pipeline, self.y_one_hot, y_as_is=self.subtask == "2")

        if self.y_padding:
            training = self.pad_X(training[0]), training[1]
            testing = self.pad_X(testing[0]), testing[1]

        return training, testing
