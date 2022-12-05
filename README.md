# NaiveBayes-TextClassification
The Naive Bayes classifier is developed to categorize documents. All the functions are written with comments to provide the intuition behind the mechanism of the Naive Bayes classifier.
This code is subject to the academic policies of DePaul University as this is part of the coursework assignment. *Therefore, use the demo code for education purposes only.*
 
### Video description can be found here:
https://youtu.be/pWJ_FQ6yE8s

### The Dataset description is given below:

The dataset is a subset of the 20 newsgroup corpus http://qwone.com/~jason/20Newsgroups/  in document-by-term format. 
This subset has been taken from http://mlg.ucd.ie/content/view/22/ (this data was modified to remove terms that did not appear in any of the documents). 

This subset includes 2,500 documents (newsgroup posts), each belonging to one of 5 categories:  comp.os.ms-windows (0), sci.crypt (1), soc.religion.christian (2), 
rec.sport.hockey (3), and misc.forsale (4). The documents are represented by 9328 terms (stems) after tokenization, stop word removal, and stemming. 
The data has been divided into test and train (20%, 80%) subsets, each presented as document-by-term matrices. The dictionary (vocabulary) for the data set is given 
in the file "newsgroup5-terms.txt" (the row index in this file corresponds to the column indexes fo the training and test matrices. 

The files contained in the zip archive are as follows:

1. newsgroup5-train.csv: the document-term frequency matrix for the training documents.
Each row of this matrix corresponds to one document and each column corresponds to one term and the (i,j)th element of 
the matrix shows the raw frequency of the jth term in the ith document. This matrix contains 2000 rows and 9328 columns.

2. newsgroup5-test.csv: the document-term frequency matrix for the test documents. The matrix contains 500 rows and 9328 columns.

3. newsgroup5-train-labels.csv: This file contains the category/class labels associated with each training document.
Each line (excluding the first line containing the label names) corresponds to a document indexed in the range of [0,2000) and
contains the numeric class label (between 0 and 4) for that document. 

4. newsgroup5-test-labels.csv: Similar to the training labels, but in this case the lines contain class labels for the 500 test
documents (excluding the first line containing the label names). 

5. newsgroup5-terms.txt: This file contains the set of 9328 terms in the vocabulary. Each line contains a term and corresponds
to the corresponding columns in training and test document-term frequency matrices. 
