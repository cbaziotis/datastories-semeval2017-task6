from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
from kutilities.callbacks import MetricsCallback, WeightsCallback, PlottingCallback
from sklearn.metrics import accuracy_score

from dataset.Semeval_Task6_data_loader import SemEval2017Task6
from models.task6A_models import humor_RNN
from utilities.data_loader import Task6Loader, prepare_dataset, get_embeddings
from utilities.ignore_warnings import set_ignores

set_ignores()
import numpy
import pickle

numpy.random.seed(1337)  # for reproducibility
text_length = 50
TASK = "1"  # Select the Subtask (1 or 2)

# specify the word vectors file to use.
# for example, WC_CORPUS = "own.twitter" and WC_DIM = 300,
# correspond to the file "datastories.twitter.300d.txt"
WV_CORPUS = "datastories.twitter"
WV_DIM = 300

# Flag that sets the training mode.
# - if FINAL == False,  then the dataset will be split in {train, val, test}
# - if FINAL == True,   then the dataset will be split in {train, val}.
# Even for training the model for the final submission a small percentage
# of the labeled data will be kept for as a validation set for early stopping
FINAL = True

# If True, the SemEval gold labels will be used as the testing set in order to perform Post-mortem analysis
SEMEVAL_GOLD = False

############################################################################
# PERSISTENCE
############################################################################
# if True save model checkpoints, as well as the corresponding word indices
# you HAVE tp set PERSIST = True, in order to be able to use the trained model later
PERSIST = False
best_model = lambda: "cp_model_task6_sub{}.hdf5".format(TASK)
best_model_word_indices = lambda: "cp_model_task6_sub{}_word_indices.pickle".format(TASK)

############################################################################
# LOAD DATA
############################################################################
embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

if PERSIST:
    pickle.dump(word_indices, open(best_model_word_indices(), 'wb'))

loader = Task6Loader(word_indices, text_lengths=(text_length, text_length), subtask=TASK, y_one_hot=False,
                     own_vectors=WV_CORPUS.startswith("own"))

if FINAL:
    print("\n > running in FINAL mode!\n")
    training, testing = loader.load_final()
else:
    training, validation, testing = loader.load_train_val_test()

if SEMEVAL_GOLD:
    print("\n > running in Post-Mortem mode!\n")
    gold_data = SemEval2017Task6().get_gold_data_task_1()
    gold_data = [v for k, v in sorted(gold_data.items())]
    X = [x for hashtag in gold_data for x in hashtag[0]]
    y = [x for hashtag in gold_data for x in hashtag[1]]
    gold = prepare_dataset(X, y, loader.pipeline, loader.y_one_hot, y_as_is=loader.subtask == "2")

    validation = testing
    testing = gold
    FINAL = False

print("Building NN Model...")
nn_model = humor_RNN(embeddings, text_length)
# nn_model = humor_CNN(embeddings, text_length)
# nn_model = humor_FFNN(embeddings, text_length)
plot(nn_model, show_layer_names=True, show_shapes=True, to_file="model_task6_sub{}.png".format(TASK))
print(nn_model.summary())

############################################################################
# CALLBACKS
############################################################################
metrics = {
    "acc": (lambda y_test, y_pred: accuracy_score(y_test, y_pred)),
}

_callbacks = []

_datasets = {}
_datasets["1-train"] = (training[0], training[1])
_datasets["2-val"] = (validation[0], validation[1]) if not FINAL else (testing[0], testing[1])
if not FINAL:
    _datasets["3-test"] = (testing[0], testing[1]) if not FINAL else None

metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)

_callbacks.append(metrics_callback)
_callbacks.append(PlottingCallback(grid_ranges=(0.5, 1.), height=4))
_callbacks.append(WeightsCallback(parameters=["W"], stats=["raster", "max", "mean", "std"]))

if PERSIST:
    checkpointer = ModelCheckpoint(filepath=best_model(), monitor='val.acc', mode="max", verbose=1, save_best_only=True)
    _callbacks.append(checkpointer)

history = nn_model.fit(training[0], training[1],
                       validation_data=(validation[0], validation[1]) if not FINAL else (testing[0], testing[1]),
                       nb_epoch=15, batch_size=256,
                       verbose=1,
                       callbacks=_callbacks)

pickle.dump(history.history, open("hist.pickle", "wb"))
