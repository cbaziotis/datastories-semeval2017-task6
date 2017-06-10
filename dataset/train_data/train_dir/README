About this folder
-----------------
This folder contains a directory that conatins the training data, which consists of 101 hashtag files.


Trial and Training Data
-----------------------

The training/trial data consits of a single directory with several files. Each file corresponds to a single hashtag, and is named appropriately. For example, for the hashtag #FastFoodBooks, the file is called Fast_Food_Books.tsv. We add the underscore between hashtag tokens for easier parsing of the hahstags. We believe a better semantic understanding of the hashtag will contribute to a better performance in the task.

The tweets are labeled 0, 1, or 2. 0 corresponds to a tweet not in the top 10 (most of the tweets in a file). 1 corresponds to a tweet in the top 10, but not the winning tweet. 2 corresponds to the winning tweet. All files will have a single tweet labeled '2'. 90 out of 101 files (in the training data dir) have the full complement of 9 tweets labeled '1'. However, the following files only have 8 such tweets:
Mom_Songs
My_Summer_Plans
Prom_In_3_Words
Make_A_Quote_Dirty
Drunk_Books
Sexy_Holidays
Marriage_Advice_In_3_Words
Best_Weekend_In_5_Words
Before_You_Tube
Sexy_Star_Wars
Hit_On_Your_Mom

We gaurentee that the evaluation data will have the appropriate amount of tweets per label.


IMPORTANT: The trial and training data is non-overlapping. Therefore, combine the data released as trial data with the data released as training data to have all available data.


Data Format
----------

The hashtag files contain three tab-separated columns:
tweet_id tweet_text tweet_label


Sample Script
------------

We released a sample script with the trial data. Information regarding it is below:

Dependencies:
Numpy
Scikit-learn

The sample script gives an idea of how to use the trial/training data. The sample script performs leave-one-out experiments on the hashtag files in the directory given as an argument to the script. Example usage:

$ python htw_sample_script.py trial_data

Specifically, the script holds-out one hashtag file at a time out. It then forms appropraite tweet pairs within the remaining (training) hashtag files. Indiviadual tweet representation is BOW frequency. The label applied to a tweet pair corresponds to whether or not the first tweet in the pair is the funnier tweet. The ordering of the pairs is random, and is chosen by a coin-flip. These pairs are then combined across all the training files to create the training matrix. An SVM model is trained on the resulting training matrix. Tweet pairs are also formed from the held-out hashtag file, and accuracy is computed on the resulting test matrix. The script reports micro-avergae accruacry across all held-out files, since different files have different amounts of tweets.


Evaluation data
---------------

For evaluation, tweets with different labels will be paired, and the goal will be to determine which tweet is the funnier. We ask that participants do not use the knowledge of label distributions directly when creating their systems. Evaluation will take place on previously unreleased hashtag files. Therefore an effective system will be able to generalize to new hashtags. We recommend to perform leave-one-out evaluation of the training files to determine the overall performance of a given system.

We are currently in the process of formaulating a second subtask that will be run on the same evaluation scheme but effectively with a different scoring method. The specific details will be available no later than mid-September.
