import pandas as pd
import numpy as np
import math, warnings


class NBClassifier:
    def __init__(self, train_csv, test_csv, vocabulary, train_df):

        self.train_lbl = pd.DataFrame(train_csv.iloc[1:, 0].values.tolist(), columns=["class"])
        self.test_lbl = pd.DataFrame(test_csv.iloc[1:, 0].values.tolist(), columns=["class"])

        self.vocabulary = vocabulary
        self.vocabulary.columns = ["term"]
        self.list_vocab = self.vocabulary["term"].values.tolist()

        self.train_df = train_df
        self.train_df.columns = self.vocabulary["term"].values.tolist()
        self.train_df["Category_label"] = self.train_lbl

        # self.test_df = test_df
        # self.test_df.columns = self.vocabulary["term"].values.tolist()
        # self.test_df["Category_label"] = self.test_lbl

        self.classes = np.array([])
        self.P_class = {}

        self.class_list()

    # This method determines the classes. Both training and testing classes are considered. It is highly unlike
    # that testing set include a class which is not include in the training set. However, if that the case, all classes
    # are determined.
    def class_list(self):
        class_tr = pd.unique(self.train_lbl["class"])
        class_ts = pd.unique(self.test_lbl["class"])
        if len(class_ts) > len(class_tr):
            print("Class imbalance in training and testing data sets. Testing data set include a class which"
                  "is excluded in training data set.")
            warnings.warn("Class imbalance between Training and Testing data sets")
        total_class = np.concatenate((class_tr, class_ts))
        self.classes = sorted(pd.unique(total_class))
        return self.classes

    def NB_train(self):

        P_class = self.train_lbl["class"].value_counts().to_dict()  # obtain the unique value counts
        total_docs = sum(P_class.values())  # compute the total count
        print("class prior probabilities")
        for i in sorted(P_class.keys()):
            self.P_class[i] = [float(P_class[i]/total_docs)]  # class probability
            print("Class ", i, ": ", float(P_class[i]/total_docs))
            tDoc = self.train_df[(self.train_df["Category_label"] == i)]  # extract the c-documents
            Tfreq = tDoc.iloc[:, :-1].sum(axis=0)  # sum of each term occurrence
            TotalTerm = Tfreq.sum()  # total term occurrence

            # compute the likely hood of each term.
            liklyhood = list(map(lambda x: ((x + 1) / (len(self.vocabulary) + TotalTerm)), Tfreq.values.tolist()))

            # append the likelyhood of each term to the class dictionary
            self.P_class[i].append(liklyhood)
        print('\n')
        return self.P_class

    def NB_test(self, test_df, test_lbl):
        length = len(test_df)
        results = []
        for i in range(length):
            sample = test_df.iloc[i].tolist()
            current_max_val = float('-inf')
            current_max_clz = ''
            for c in self.P_class.keys():
                # natural log of each likelihood of term is multiply with the frequency of corresponding term at
                # given test instance. map() function is used to compute it.
                maxv = self.P_class[c][0] + sum(list(map(lambda x, y: math.log(x)*y, self.P_class[c][1], sample)))
                if current_max_val < maxv:
                    current_max_val = maxv
                    current_max_clz = c
            results.append([i, int(test_lbl.iloc[i][0]), int(current_max_clz), current_max_val])
        return results

    # calculating the posterior probability of a term in the each class
    def posterior_feature(self, P_class, vocab, term):
        # if the term is a single word
        if type(term) == str:
            indx = vocab.index(term)
            posterior = []
            print("Posteriors of term:", term, "\n")
            for c in P_class.keys():
                posterior.append(P_class[c][1][indx])
                print("Class label ", c, " : ", str(P_class[c][1][indx]))
            print("\n")

        # if the term is a list of words
        if type(term) == list:
            posterior = []
            for t in term:
                indx = vocab.index(t)
                temp = []
                print("Posteriors of term:", t, "\n")
                for c in P_class.keys():
                    temp.append(P_class[c][1][indx])
                    print("Class label ", c, " : ", str(P_class[c][1][indx]))
                posterior.append([t, temp])
                print("\n")
        return posterior

    def accuracy(self, results, printlength):
        length = len(results)
        correct = 0
        printF = False
        if len(printlength) != 0:
            printF = True
        for i in range(length):
            if results[i][1] == results[i][2]:
                correct += 1
            if printF and (printlength[0] <= i < printlength[1]):
                print("Test Item ", results[i][0], " - Predicted class: "
                      , results[i][2], ", Actual class: ", results[i][1], ", Log probability: ", results[i][3])
        accuracy = correct/length * 100
        print("\n")
        print("Accuracy: {:2f}%".format(accuracy))
        return accuracy
