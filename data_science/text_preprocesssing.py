# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import spacy 
from nltk.corpus import stopwords
import string
import re
import itertools
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.en import English
parser = English()
from gensim.models import Phrases, Word2Vec, Doc2Vec
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt


def spacify(message): 
    
    tokens = parser(unicode(message, 'utf-8', errors='replace'))
    list_of_tokens = [token.lemma_ for token in tokens if token.is_punct == False and token.is_stop == False and
          token.is_digit == False]
    return list_of_tokens

def cost_matrix(confusion_matrix):
    cost_array = np.array([[1, 1, -10, -100, -500], [1, 1, -5, -50, -500], [-25, -5, 1, -25, -500], [-50, -25, -5, 1, -100], 
[-100, -50, -25, -5, 1]])
    return confusion_matrix * cost_array

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]), decimals = 3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 1.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca", "'d", "'ll", "'ve", "'re", "'r", "ed"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”"]

# Every step in a pipeline needs to be a "transformer". 
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")

    
    # replace HTML symbols
    text = text.replace("&copy;", " ").replace("&reg;", " ").replace("&yen;	", " ")
    text = text.replace("&amp;", "and").replace("&gt;", " ").replace("&lt;", " ").replace("&gt;", " ").replace("&pound;", " ").replace("&euro;", " ")
    
    # lemmatize
    text = text.lower()

    #Change to unicode for Spacy
    #text = unicode(text, 'utf-8', errors='ignore')
    
    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(text):

    # get the tokens using spaCy
    #tokens = parser(unicode(text, 'utf-8', errors='ignore'))
    tokens = parser(text)
    # lemmatize
    lemmas = [tok.lemma_.lower().strip() for tok in tokens if not (tok.lemma_ == "-PRON-"  or tok.is_oov or tok.is_punct or tok.like_num or tok.like_email or tok.like_url or tok.is_stop or tok.is_space)]
    tokens = lemmas
    

    # stoplist the tokens,symbols and take anything longer than one character
    tokens = [tok for tok in tokens if tok not in (STOPLIST and SYMBOLS) and len(tok)> 1]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens: 
        tokens.remove("\n\n")
    while "\n\n\n" in tokens:
        tokens.remove("\n\n\n")
        
    return tokens

def bi_tri_grammize(tokens):
    gram_model = Phrases(tokens)
    grammized_text = []
    for i in tokens:
        grammized_text.append(gram_model[i])
    return grammized_text
    
def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)
