'''
This module is used for performing a Leave-One-Out validation on the dataset,
in order to get comparable results with the Potash et al. "# HashtagWars: Learning a Sense of Humor." arXiv:1612.03216.
'''
from keras.callbacks import EarlyStopping
from kutilities.callbacks import MetricsCallback
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

from models.task6A_models import humor_RNN
from utilities.data_loader import get_embeddings, Task6Loader
from utilities.ignore_warnings import set_ignores

set_ignores()
import numpy
import pickle
import os


numpy.random.seed(1337)  # for reproducibility
text_length = 50
TASK = "1"  # Select the Subtask (1 or 2)
WV_CORPUS = "datastories.twitter"
WV_DIM = 300

FINAL = True

############################################################################
# LOAD DATA
############################################################################
embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

loader = Task6Loader(word_indices, text_lengths=(text_length, text_length), subtask=TASK, y_one_hot=False,
                     own_vectors=WV_CORPUS.startswith("own"))

data, hashtags = loader.load_loo()

############################################################################
# CALLBACKS
############################################################################
metrics = {
    "acc": (lambda y_test, y_pred: accuracy_score(y_test, y_pred)),
}


def eval_run(training, testing):
    metrics_callback = MetricsCallback(
        training_data=(training[0], training[1]),
        validation_data=(testing[0], testing[1]),
        metrics=metrics)
    early_stopping = EarlyStopping(monitor='val.acc', mode="max", patience=3)

    print("Building NN Model...")
    nn_model = humor_RNN(embeddings, text_length)
    history = nn_model.fit(training[0], training[1],
                           validation_data=(testing[0], testing[1]),
                           nb_epoch=1, batch_size=256,
                           callbacks=[metrics_callback, early_stopping])
    return history.history


def loo_run(run):
    print("\n\n\n------------------\nRUN={}\n------------------".format(run))
    loo = LeaveOneOut()
    os.system('cls')
    results = []
    for train_index, test_index in loo.split(data):
        train, test, hashtag = data[train_index], data[test_index], hashtags[test_index]

        X_train = [numpy.asarray([x for hashtag in train for x in hashtag[0][0]]),
                   numpy.asarray([x for hashtag in train for x in hashtag[0][1]])]
        y_train = numpy.concatenate([x[1] for x in train], axis=0)

        X_test = [numpy.asarray([x for hashtag in test for x in hashtag[0][0]]),
                  numpy.asarray([x for hashtag in test for x in hashtag[0][1]])]
        y_test = numpy.concatenate([x[1] for x in test], axis=0)

        hashtag = hashtag[0]
        print("\n------------------\nHASHTAG={}\n------------------".format(hashtag))
        results.append((hashtag, eval_run((X_train, y_train), (X_test, y_test))))
        pickle.dump(results, open("results\\results_loo_{}.pickle".format(str(run)), "wb"))


# perform 3 LOO runs, and dump to disk the results of each one
for i in range(3):
    loo_run(i + 1)
