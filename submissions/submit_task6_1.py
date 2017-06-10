"""
Created by Christos Baziotis.
"""
import pickle

import numpy
from keras.models import load_model
from keras.utils.visualize_util import plot
from kutilities.layers import AttentionWithContext, MeanOverTime, Attention
from sklearn.metrics import accuracy_score

from dataset.Semeval_Task6_data_loader import SemEval2017Task6
from utilities.data_loader import Task6Loader, prepare_text_only_dataset, prepare_dataset

numpy.random.seed(1337)  # for reproducibility
text_length = 50
TASK = "1"
WC_CORPUS = "datastories.twitter"
WC_DIM = 200

best_model = "cp_model_task6_sub{}.hdf5".format(TASK)
best_model_word_indices = "cp_model_task6_sub{}_word_indices.pickle".format(TASK)

print("Loading model from disk...", end=" ")
model = load_model(best_model, custom_objects={'AttentionWithContext': AttentionWithContext,
                                               'MeanOverTime': MeanOverTime,
                                               'Attention': Attention})

plot(model, show_layer_names=True, show_shapes=True, to_file="model_task6_sub{}.png".format(TASK))
print(model.summary())

print("done!")
print("Loading word indices from disk...", end=" ")
word_indices = pickle.load(open(best_model_word_indices, "rb"))
print("done!")
print("Loading Word Embeddings from disk...", end=" ")
loader = Task6Loader(word_indices, text_lengths=(text_length, text_length), subtask=TASK, y_one_hot=False,
                     own_vectors=WC_CORPUS.startswith("own"))
print("done!")
print("Loading test data...")
test_data = SemEval2017Task6().get_test_data_task_1()
for hfile, pairs in test_data.items():
    X = prepare_text_only_dataset([x[1] for x in pairs], loader.pipeline)

    print("Making predictions...")
    y_pred = model.predict(X, batch_size=2048, verbose=0)
    y_pred = numpy.array([int(_y > 0.5) for _y in y_pred])

    submit_fname = hfile.split(".")[0] + "_PREDICT.tsv"
    with open(submit_fname, "w") as f:
        for pair, prediction in zip(pairs, y_pred):
            f.write("\t".join([pair[0][0], pair[0][1], str(prediction)]) + "\n")

##############################################################
# Classification

print("DOUBLE CHECK")
gold_data = SemEval2017Task6().get_gold_data_task_1()
gold_data = [v for k, v in sorted(gold_data.items())]
X = [x for hashtag in gold_data for x in hashtag[0]]
y = [x for hashtag in gold_data for x in hashtag[1]]
gold = prepare_dataset(X, y, loader.pipeline, loader.y_one_hot, y_as_is=loader.subtask == "2")

##############################################################
# Classification

print("PARANOID CHECK")
training, testing = loader.load_final()


def make_predictions(_X, _y):
    y_pred = model.predict(_X, batch_size=2048, verbose=0)
    y_pred = numpy.array([int(_y > 0.5) for _y in y_pred])
    print("accuracy", accuracy_score(_y, y_pred))


print("Making predictions training...")
make_predictions(training[0], training[1])

print("Making predictions testing...")
make_predictions(testing[0], testing[1])

print("Making predictions gold...")
make_predictions(gold[0], gold[1])
