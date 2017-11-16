from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import sys
import sklearn

# COMMENT THIS
# Read in the corpus file (trainset.txt) line by line, append only the text to the documents list.
# If use_sentiment == 1, use pos vs neg labels, else use the 6-class labels (books, camera, dvd, health, music, software)
# Return the list with all document content and corresponding labels
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

def printErrorMSG():
    print('Please provide 1 argument: True for sentiment testing and False for category testing',file=sys.stderr)

def customEvaluation(Ytest, Yguess, use_sentiment):
    #scores per class
    labels_sentiment = ["pos","neg"]
    labels_categories = ["books","health","dvd","camera","music","software"]
    labels = labels_sentiment if use_sentiment else labels_categories
    print('{:<15}{:<15}{:<15}{:<15}'.format('Class', 'Precision', 'Recall', 'F-score'))
    for label in labels:
        precisionScore = sklearn.metrics.precision_score(Ytest,Yguess, average="macro", labels=label)
        recallScore = sklearn.metrics.recall_score(Ytest,Yguess, average="macro", labels=label)
        f1Score = sklearn.metrics.f1_score(Ytest,Yguess, average="macro", labels=label)
        print('{:<15}{:<15}{:<15}{:<15}'.format(label, round(precisionScore,3), round(recallScore,3), round(f1Score,3)))

def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == 'True':
            condition = True
        elif sys.argv[1] == 'False':
            condition = False
        else:
            printErrorMSG()
        
        # COMMENT THIS
        # The data is split in train/test data, 75% train and 25% test. 
        X, Y = read_corpus('trainset.txt', use_sentiment=condition)
        split_point = int(0.75*len(X))
        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]
        Xtest = X[split_point:]
        Ytest = Y[split_point:]

        # let's use the TF-IDF vectorizer
        tfidf = True

        # we use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenized.
        if tfidf:
            vec = TfidfVectorizer(preprocessor = identity,
                                  tokenizer = identity)
        else:
            vec = CountVectorizer(preprocessor = identity,
                                  tokenizer = identity)

        # combine the vectorizer with a Naive Bayes classifier
        classifier = Pipeline( [('vec', vec),
                                ('cls', MultinomialNB())] )


        # COMMENT THIS
        # The classifier learns from the model / is trained (classifier is trained on the test data)
        classifier.fit(Xtrain, Ytrain)

        # COMMENT THIS  
        # Values are predicted (test data input, predicted results output)
        Yguess = classifier.predict(Xtest)

        # COMMENT THIS
        # Accuracy is printed, predicted labels are compared to the original labels and percentage correct is given
        print(accuracy_score(Ytest, Yguess))

        customEvaluation(Ytest, Yguess, use_sentiment=condition)

    else:
        printErrorMSG() 

main()
