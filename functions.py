'''
--------------------------------------------------------------------------------
This is a Python script to collect news articles for the final project for
Computational Content Analysis course in Winter 2017.
--------------------------------------------------------------------------------
This script defines the following functions needed to perform content analysis:
    * normalizeTokens
    * dropMissing (From CCA Notebook 3: Clustering and Topic Modeling)
    * normalize (From CCA Notebook 4: Word Embedding)
    * dimension (From CCA Notebook 4: Word Embedding)
    * makeDF (From CCA Notebook 4: Word Embedding)
    * coloring (From CCA Notebook 4: Word Embedding)
    * PlotDimension (From CCA Notebook 4: Word Embedding)
--------------------------------------------------------------------------------
'''
#All these packages need to be installed from pip
#These are all for the cluster detection
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import scipy #For hierarchical clustering and some visuals
#import scipy.cluster.hierarchy
import gensim#For topic modeling
import nltk #the Natural Language Toolkit
import requests #For downloading our datasets
import numpy as np #for arrays
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import matplotlib.cm #Still for graphics
import seaborn as sns #Makes the graphics look nicer

import json
import bs4

'''--------------------------------------------------------------------------'''
def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
    #We will return a list with the stopwords removed
    return list(workingIter)

'''--------------------------------------------------------------------------'''
def dropMissing(wordLst, vocab):
    return [w for w in wordLst if w in vocab]

'''--------------------------------------------------------------------------'''
def normalize(vector):
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector

'''--------------------------------------------------------------------------'''
def dimension(model, positives, negatives):
    diff = sum([normalize(model[x]) for x in positives]) - sum([normalize(model[y]) for y in negatives])
    return diff

'''--------------------------------------------------------------------------'''
def makeDF(model, word_list, dimension1, dimension2, dimension3):
    dim1 = []
    dim2 = []
    dim3 = []
    for word in word_list:
        dim1.append(sklearn.metrics.pairwise.cosine_similarity(model[word].reshape(1,-1), dimension1.reshape(1,-1))[0][0])
        dim2.append(sklearn.metrics.pairwise.cosine_similarity(model[word].reshape(1,-1), dimension2.reshape(1,-1))[0][0])
        dim3.append(sklearn.metrics.pairwise.cosine_similarity(model[word].reshape(1,-1), dimension3.reshape(1,-1))[0][0])
    df = pandas.DataFrame({'Dimension1': dim1, 'Dimension2': dim2, 'Dimension3': dim3}, index = word_list)
    return df

'''--------------------------------------------------------------------------'''
def Coloring(Series):
    x = Series.values
    y = x-x.min()
    z = y/y.max()
    c = list(plt.cm.rainbow(z))
    return c

'''--------------------------------------------------------------------------'''
def PlotDimension(ax, df, dim):
    ax.set_frame_on(False)
    ax.set_title(dim, fontsize = 18)
    colors = Coloring(df[dim])
    for i, word in enumerate(df.index):
        ax.annotate(word, (0, df[dim][i]), color = colors[i], alpha = 0.6, fontsize = 15)
    MaxY = df[dim].max()
    MinY = df[dim].min()
    plt.ylim(MinY,MaxY)
    plt.yticks(())
    plt.xticks(())
