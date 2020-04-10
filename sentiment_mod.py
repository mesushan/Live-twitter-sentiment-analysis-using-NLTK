import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


#creating the VoteClassifier class
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
import codecs

short_pos = codecs.open("short_reviews/positive.txt","r", encoding="latin2").read()
short_neg = codecs.open("short_reviews/negative.txt","r", encoding="latin2").read()


#creating empty lists
all_words = []
documents = []

#only allowing the adjective word type
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

#loading the pickled file
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

#frequency distribution to find the most common words
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))

#using top 5000 features or words (can be used any features accordingly)
word_features = list(all_words.keys())[:5000]


#loading the pickled file
word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


#function to find the features in the document
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) #if the 5000 words in doc, it will be true else false

    return features


#loading the pickled file
featuresets_f = open("pickled_algos/featuresets5k.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()


#shuffling the featuresets 
random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


#loading the pickled classifier
open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


#loading the pickled classifier
open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


#loading the pickled classifier
open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


#loading the pickled classifier
open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


#loading the pickled classifier
open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


#loading the pickled classifier
open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


#creating a voting classifier which trains on an ensemble of all the above classifiers 
voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


#function to utilize the voted classifier and predict outcomes
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)