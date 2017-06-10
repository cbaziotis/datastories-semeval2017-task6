This repository contains the source code for the models used for _DataStories_ team's submission 
for SemEval-2017 Task 6 “#HashtagWars: Learning a Sense of Humor”.


## Prerequisites
#### 1 - Install Requirements
```
pip install -r /datastories-semeval2017-task6/requirements.txt
```

#### 2 - Download pre-trained Word Embeddings
The models were trained on top of word embeddings pre-trained on a big collection of Twitter messages that we collected.
We collected a big dataset of 330M English Twitter messages posted from 12/2012 to 07/2016. 
For training the word embeddings we used [GloVe](https://github.com/stanfordnlp/GloVe).
For preprocessing the tweets we used [ekphrasis](https://github.com/cbaziotis/ekphrasis), 
which is also one of the requirements of this project.

You can download one of the following word embeddings:
- [datastories.twitter.50d.txt](https://github.com/cbaziotis/ekphrasis): 50 dimensional embeddings
- [datastories.twitter.100d.txt](https://github.com/cbaziotis/ekphrasis): 100 dimensional embeddings
- [datastories.twitter.200d.txt](https://github.com/cbaziotis/ekphrasis): 200 dimensional embeddings
- [datastories.twitter.300d.txt](https://github.com/cbaziotis/ekphrasis): 300 dimensional embeddings

Place the file(s) in `/embeddings` folder, for the program to find it.


## Overview

This project contains the complete source code for training a NN model 
for SemEval-2017 Task 6 “#HashtagWars: Learning a Sense of Humor” and 
the trained model that was used for our final submission. 

\* If what you are interested in is just the source code for the model then just see 
[models/task6A_models.py](https://github.com/cbaziotis/datastories-semeval2017-task6/blob/master/models/task6A_models.py).



#### Word Embeddings
In order to specify which word embeddings file you want to use, 
you have to set the values of `WV_CORPUS` and `WV_WV_DIM` in `task6A.py` and `task6A_LOO.py` respectively.
In our submission we used:
```python
WV_CORPUS = "datastories.twitter"
WV_DIM = 300
```

The convention we use to identify each file is:
```
{corpus}.{dimensions}d.txt
```

This means that if you want to use another file, for instance GloVe Twitter word embeddings with 200 dimensions,
you have to place a file like `glove.200d.txt` inside `/embeddings` folder and set:
```python
WV_CORPUS = "glove"
WV_DIM = 200
```

