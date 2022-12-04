import pandas as pd
from NBClassifier import NBClassifier

train_csv = pd.read_csv("DataSet/newsgroups5-train-labels.csv", header=None)
test_csv = pd.read_csv("DataSet/newsgroups5-test-labels.csv", header=None)
vocabulary = pd.read_csv("DataSet/newsgroups5-terms.txt", header=None)
train_df = pd.read_csv("DataSet/newsgroups5-train.csv", header=None)
test_df = pd.read_csv("DataSet/newsgroups5-test.csv", header=None)

nb = NBClassifier(train_csv, test_csv, vocabulary, train_df)
clz = nb.classes
P_class = nb.NB_train()
results = nb.NB_test(test_df, nb.test_lbl)
pos = nb.posterior_feature(P_class, nb.list_vocab, ["game", "god", "match", "program", "sale"])
accuracy = nb.accuracy(results, [0, 20])





