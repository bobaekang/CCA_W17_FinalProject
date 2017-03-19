'''
--------------------------------------------------------------------------------
This is a Python script to analyze news articles for the final project for
Computational Content Analysis course in Winter 2017.
--------------------------------------------------------------------------------
This script analyzes articles from the following newspapers:
    * The Guardian (using API)
    * The Daily Mail (scraping from the web)

This script performs the following tasks:
    * Import packages and corpora
    * Clustering and topic modeling (Notebook 3)
    * Word embedding (Notebook 4)
--------------------------------------------------------------------------------
'''
# import packages
import gensim #For word2vec, etc
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA
import requests #For downloading our datasets
import nltk #For stop words and stemmers
import numpy as np #For arrays
import pandas as pd #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import os #For looking through files
import os.path #For managing file paths
import json
import bs4
import re
import random
random.seed(1234)

dailymailDF = pd.read_pickle('data/dailymail_all.pkl')
guardianDF = pd.read_pickle('data/guardian_all.pkl')

# summary of the corpus
print('-----------------------------------------------------------------------')
print('Summary of the corpus:')
print('-----------------------------------------------------------------------')
print('There are total {} articles in the Daily Mail corpus.'.format(int(dailymailDF.shape[0])))
print('{} articles in the Daily Mail corpus are in Category 0.'.format(int(dailymailDF[dailymailDF.category == '0'].shape[0])))
print('{} articles in the Daily Mail corpus are in Category 1.'.format(int(dailymailDF[dailymailDF.category == '1'].shape[0])))
print('There are total {} word tokens in the Daily Mail corpus.'.format(int(dailymailDF.token_counts.sum())))
print('There are total {} articles in the Guardian corpus.'.format(int(guardianDF.shape[0])))
print('{} articles in the Guardian corpus are in Category 0.'.format(int(guardianDF[guardianDF.category == '0'].shape[0])))
print('{} articles in the Guardian corpus are in Category 1.'.format(int(guardianDF[guardianDF.category == '1'].shape[0])))
print('There are total {} word tokens in the Guardian corpus.'.format(int(guardianDF.token_counts.sum())))
print('-----------------------------------------------------------------------')


'''
--------------------------------------------------------------------------------
Clustering (from Notebook 3)
--------------------------------------------------------------------------------
In this part, I examine each corpus using principal component analysis to ensure
that articles in different categories ('A' for automatation and other associated
keywords and 'I' for immigration and other associated keywords) are indeed
different from each other.
--------------------------------------------------------------------------------
'''
from functions import normlizeTokens, dropMissing

'''Prepare my corpora for clustering'''
# Initialize CountVectorizer(): both
CountVectorizer = sklearn.feature_extraction.text.CountVectorizer()
# train
dailymailVects = CountVectorizer.fit_transform(dailymailDF['text'])
guardianVects = CountVectorizer.fit_transform(guardianDF['text'])

# Initionalize TfidfTransformer(): dailymail
dailymailTFTransformer = sklearn.feature_extraction.text.TfidfTransformer().fit(dailymailVects)
guardianTFTransformer = sklearn.feature_extraction.text.TfidfTransformer().fit(guardianVects)
# train TFTransformer: dailymail
dailymailTF = dailymailTFTransformer.transform(dailymailVects)
guardianTF = guardianTFTransformer.transform(guardianVects)

# Initialize TfidfVectorizer(): both
TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000,
                                                            min_df=3, stop_words='english', norm='l2')
# Train Tf-idf
dailymailTFVects = TFVectorizer.fit_transform(dailymailDF['text'])
guardianTFVects = TFVectorizer.fit_transform(guardianDF['text'])

# initialize our stemmer and our stop words
stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')

# get normalized and reduced tokens
dailymailDF['normalized_tokens'] = dailymailDF['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))
dailymailDF['reduced_tokens'] = dailymailDF['normalized_tokens'].apply(lambda x: dropMissing(x, TFVectorizer.vocabulary_.keys()))
guardianDF['normalized_tokens'] = guardianDF['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))
guardianDF['reduced_tokens'] = guardianDF['normalized_tokens'].apply(lambda x: dropMissing(x, TFVectorizer.vocabulary_.keys()))


'''Perform K-means clustering'''
numCategories = len(set(dailymailDF['category']))

# Running k means: dailymail
dailymailKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
dailymailKM.fit(dailymailTFVects)
# Running k means: guardian
guardianKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
guardianKM.fit(guardianTFVects)

# contents of the clusters
terms = TFVectorizer.get_feature_names()
d_order_centroids = dailymailKM.cluster_centers_.argsort()[:, ::-1]
g_order_centroids = guardianKM.cluster_centers_.argsort()[:, ::-1]
print("Top terms per cluster--dailymail:")
for i in range(numCategories):
    print("Cluster %d:" % i)
    for ind in d_order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')
print("Top terms per cluster--guadian:")
for i in range(numCategories):
    print("Cluster %d:" % i)
    for ind in g_order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')

print('-----------------------------------------------------------------------')
print("The available metrics are: {}".format([s for s in dir(sklearn.metrics) if s[0] != '_']))
print("for our Dailymail corpus clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(dailymailDF['topic'], dailymailKM.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(dailymailDF['topic'], dailymailKM.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(dailymailDF['topic'], dailymailKM.labels_)))
print("for our Guardian corpus clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(guardianDF['topic'], guardianKM.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(guardianDF['topic'], guardianKM.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(guardianDF['topic'], guardianKM.labels_)))
print('-----------------------------------------------------------------------')

'''Visualize each corpus'''
# reduce dimensionality with PCA: dailymail
d_pca = sklearn.decomposition.PCA(n_components = 2).fit(dailymailTFVects.toarray())
d_reduced_data = d_pca.transform(dailymailTFVects.toarray())

d_components = d_pca.components_
d_keyword_ids = list(set(d_order_centroids[:,:10].flatten())) #Get the ids of the most distinguishing words(features) from your kmeans model.
dwords = [terms[i] for i in d_keyword_ids]#Turn the ids into words.
dx = d_components[:,d_keyword_ids][0,:] #Find the coordinates of those words in your biplot.
dy = d_components[:,d_keyword_ids][1,:]

# reduce dimensionality with PCA: guardian
g_pca = sklearn.decomposition.PCA(n_components = 2).fit(guardianTFVects.toarray())
g_reduced_data = g_pca.transform(guardianTFVects.toarray())

g_components = g_pca.components_
g_keyword_ids = list(set(g_order_centroids[:,:10].flatten())) #Get the ids of the most distinguishing words(features) from your kmeans model.
gwords = [terms[i] for i in g_keyword_ids]#Turn the ids into words.
gx = g_components[:,g_keyword_ids][0,:] #Find the coordinates of those words in your biplot.
gy = g_components[:,g_keyword_ids][1,:]

# color map for true categories/topics
colordict = {
'0': 'red',
'1': 'blue',
    }
dcolors = [colordict[c] for c in dailymailDF['category']] # colormap for dailymail
gcolors = [colordict[c] for c in guardianDF['category']] # colormap for guardian
print("The categories' colors are:\n{}".format(colordict.items()))

# dailymail plots
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(d_reduced_data[:, 0], d_reduced_data[:, 1], color = dcolors, alpha = 0.3, label = dcolors)
for i, word in enumerate(dwords):
    ax.annotate(word, (dx[i],dy[i]))
plt.xticks(())
plt.yticks(())
plt.title('True Classes: Daily Mail')
plt.savefig('image/figCorpusD', bbox_inches = 'tight')
plt.show()
plt.close()

# guardian plots
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(g_reduced_data[:, 0], g_reduced_data[:, 1], color = gcolors, alpha = 0.3, label = gcolors)
for i, word in enumerate(gwords):
    ax.annotate(word, (gx[i],gy[i]))
plt.xticks(())
plt.yticks(())
plt.title('True Classes: The Guardian')
plt.savefig('image/figCorpusG', bbox_inches = 'tight')
plt.show()
plt.close()

'''Visualize K-means clustering outcomes'''
dcolors_p = [colordict[['I', 'A'][l]] for l in dailymailKM.labels_]
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(d_reduced_data[:, 0], d_reduced_data[:, 1], color = dcolors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters: Daily Mail\n k = 2')
plt.show()

gcolors_p = [colordict[['A', 'I'][l]] for l in guardianKM.labels_]
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(g_reduced_data[:, 0], g_reduced_data[:, 1], color = gcolors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters: The Guardian\n k = 2')
plt.show()
'''
--------------------------------------------------------------------------------
Topic modeling (from CCA Notebook 3)
--------------------------------------------------------------------------------
In this part, I divide each corpus into two subcorpora according to categories.
Then I use topic modeling algorithm to examine the contents of each subcorpus.
--------------------------------------------------------------------------------
'''
'''Prepare my corpora for topic modeling'''
# create dictionary
dictionary = gensim.corpora.Dictionary(dailymailDF['reduced_tokens'].append(guardianDF['reduced_tokens'], ignore_index=True))

# create a corpus for each
d0_corpus = [dictionary.doc2bow(text) for text in dailymailDF[dailymailDF.category == '0']['reduced_tokens']]
d1_corpus = [dictionary.doc2bow(text) for text in dailymailDF[dailymailDF.category == '1']['reduced_tokens']]
g0_corpus = [dictionary.doc2bow(text) for text in guardianDF[guardianDF.category == '0']['reduced_tokens']]
g1_corpus = [dictionary.doc2bow(text) for text in guardianDF[guardianDF.category == '1']['reduced_tokens']]

# serialize
# gensim.corpora.MmCorpus.serialize('data/dailymail0.mm', d0_corpus)
# gensim.corpora.MmCorpus.serialize('data/dailymail1.mm', d1_corpus)
# gensim.corpora.MmCorpus.serialize('data/guardian0.mm', g0_corpus)
# gensim.corpora.MmCorpus.serialize('data/guardian1.mm', g1_corpus)
# load serialzed corpora
d0_mm = gensim.corpora.MmCorpus('data/dailymail0.mm')
d1_mm = gensim.corpora.MmCorpus('data/dailymail1.mm')
g0_mm = gensim.corpora.MmCorpus('data/guardian0.mm')
g1_mm = gensim.corpora.MmCorpus('data/guardian1.mm')


'''Perform topic modeling'''
# train lda models
# d0_lda = gensim.models.ldamodel.LdaModel(corpus=d0_mm, id2word=dictionary, num_topics=5, alpha='auto', eta='auto')
# d1_lda = gensim.models.ldamodel.LdaModel(corpus=d1_mm, id2word=dictionary, num_topics=5, alpha='auto', eta='auto')
# g0_lda = gensim.models.ldamodel.LdaModel(corpus=g0_mm, id2word=dictionary, num_topics=5, alpha='auto', eta='auto')
# g1_lda = gensim.models.ldamodel.LdaModel(corpus=g1_mm, id2word=dictionary, num_topics=5, alpha='auto', eta='auto')
# save trained models
# d0_lda.save('model/d0_lda.model')
# d1_lda.save('model/d1_lda.model')
# g0_lda.save('model/g0_lda.model')
# g1_lda.save('model/g1_lda.model')
# load trained models
d0_lda = gensim.models.ldamodel.LdaModel.load('model/d0_lda.model')
d1_lda = gensim.models.ldamodel.LdaModel.load('model/d1_lda.model')
g0_lda = gensim.models.ldamodel.LdaModel.load('model/g0_lda.model')
g1_lda = gensim.models.ldamodel.LdaModel.load('model/g1_lda.model')


# to dataframe with topics
d0_ldaDF = pd.DataFrame({
        'url' : dailymailDF[dailymailDF.category == '0']['url'],
        'topics' : [d0_lda[dictionary.doc2bow(l)] for l in dailymailDF[dailymailDF.category == '0']['reduced_tokens']]
    })
d1_ldaDF = pd.DataFrame({
        'url' : dailymailDF[dailymailDF.category == '1']['url'],
        'topics' : [d1_lda[dictionary.doc2bow(l)] for l in dailymailDF[dailymailDF.category == '1']['reduced_tokens']]
    })

#Dict to temporally hold the probabilities
d0_topicsProbDict = {i : [0] * len(d0_ldaDF) for i in range(d0_lda.num_topics)}
d1_topicsProbDict = {i : [0] * len(d1_ldaDF) for i in range(d1_lda.num_topics)}
# d_topicsProbDict = {i : [0] * len(d_ldaDF) for i in range(d_lda.num_topics)}

#Load them into the dict
for index, topicTuples in enumerate(d0_ldaDF['topics']):
    for topicNum, prob in topicTuples:
        d0_topicsProbDict[topicNum][index] = prob
for index, topicTuples in enumerate(d1_ldaDF['topics']):
    for topicNum, prob in topicTuples:
        d1_topicsProbDict[topicNum][index] = prob

#Update the DataFrame
for topicNum in range(d0_lda.num_topics):
    d0_ldaDF['topic_{}'.format(topicNum)] = d0_topicsProbDict[topicNum]
for topicNum in range(d1_lda.num_topics):
    d1_ldaDF['topic_{}'.format(topicNum)] = d1_topicsProbDict[topicNum]

# to dataframe with topics
g0_ldaDF = pd.DataFrame({
        'url' : guardianDF[guardianDF.category == '0']['url'],
        'topics' : [g0_lda[dictionary.doc2bow(l)] for l in guardianDF[guardianDF.category == '0']['reduced_tokens']]
    })
g1_ldaDF = pd.DataFrame({
        'url' : guardianDF[guardianDF.category == '1']['url'],
        'topics' : [g1_lda[dictionary.doc2bow(l)] for l in guardianDF[guardianDF.category == '1']['reduced_tokens']]
    })

#Dict to temporally hold the probabilities
g0_topicsProbDict = {i : [0] * len(g0_ldaDF) for i in range(g0_lda.num_topics)}
g1_topicsProbDict = {i : [0] * len(g1_ldaDF) for i in range(g1_lda.num_topics)}
# g_topicsProbDict = {i : [0] * len(g_ldaDF) for i in range(g_lda.num_topics)}

#Load them into the dict
for index, topicTuples in enumerate(g0_ldaDF['topics']):
    for topicNum, prob in topicTuples:
        g0_topicsProbDict[topicNum][index] = prob
for index, topicTuples in enumerate(g1_ldaDF['topics']):
    for topicNum, prob in topicTuples:
        g1_topicsProbDict[topicNum][index] = prob

#Update the DataFrame
for topicNum in range(g0_lda.num_topics):
    g0_ldaDF['topic_{}'.format(topicNum)] = g0_topicsProbDict[topicNum]
for topicNum in range(g1_lda.num_topics):
    g1_ldaDF['topic_{}'.format(topicNum)] = g1_topicsProbDict[topicNum]


'''Examine topics'''
# topics--dailymail, category = 0
d0_topicsDict = {}
for topicNum in range(d0_lda.num_topics):
    topicWords = [w for w, p in d0_lda.show_topic(topicNum)]
    d0_topicsDict['Topic_{}'.format(topicNum)] = topicWords
d0_wordRanksDF = pd.DataFrame(d0_topicsDict)
d0_wordRanksDF

# topics--dailymail, category = 1
d1_topicsDict = {}
for topicNum in range(d1_lda.num_topics):
    topicWords = [w for w, p in d1_lda.show_topic(topicNum)]
    d1_topicsDict['Topic_{}'.format(topicNum)] = topicWords
d1_wordRanksDF = pd.DataFrame(d1_topicsDict)
d1_wordRanksDF

# topics--guardian, category = 0
g0_topicsDict = {}
for topicNum in range(g0_lda.num_topics):
    topicWords = [w for w, p in g0_lda.show_topic(topicNum)]
    g0_topicsDict['Topic_{}'.format(topicNum)] = topicWords
g0_wordRanksDF = pd.DataFrame(g0_topicsDict)
g0_wordRanksDF

# topics--guardian, category = 1
g1_topicsDict = {}
for topicNum in range(g1_lda.num_topics):
    topicWords = [w for w, p in g1_lda.show_topic(topicNum)]
    g1_topicsDict['Topic_{}'.format(topicNum)] = topicWords
g1_wordRanksDF = pd.DataFrame(g1_topicsDict)
g1_wordRanksDF

print('-----------------------------------------------------------------------')
print('Topic Modeling results')
print('-----------------------------------------------------------------------')
print('Topics: Daily Mail, Category 0')
print(d0_wordRanksDF)
print('Topics: Daily Mail, Category 1')
print(d1_wordRanksDF)
print('-----------------------------------------------------------------------')
print('Topics: Guardian, Category 0')
print(g0_wordRanksDF)
print('Topics: Guardian, Category 1')
print(g1_wordRanksDF)
print('-----------------------------------------------------------------------')

'''Visualize topic modeling'''
d0_ldaDFV = d0_ldaDF[:10][['topic_%d' %x for x in range(10)]]
d0_ldaDFVisN = d0_ldaDF[:10][['url']]
d0_ldaDFVis = d0_ldaDFV.as_matrix(columns=None)
d1_ldaDFV = d1_ldaDF[:10][['topic_%d' %x for x in range(10)]]
d1_ldaDFVisN = d1_ldaDF[:10][['url']]
d1_ldaDFVis = d1_ldaDFV.as_matrix(columns=None)

# extract 'title' part from each url
for i in range(10):
    d0_name = re.search('article.*/(.*)\.html', d0_ldaDFVisN.url.iloc[i]).group(1)
    d0_ldaDFVisN.url.iloc[i] = d0_name
    d1_name = re.search('article.*/(.*)\.html', d1_ldaDFVisN.url.iloc[i]).group(1)
    d1_ldaDFVisN.url.iloc[i] = d1_name
d0_ldaDFVisNames = d0_ldaDFVisN.as_matrix(columns=None)
d1_ldaDFVisNames = d1_ldaDFVisN.as_matrix(columns=None)

g0_ldaDFV = g0_ldaDF[:10][['topic_%d' %x for x in range(10)]]
g0_ldaDFVisN = g0_ldaDF[:10][['url']]
g0_ldaDFVis = g0_ldaDFV.as_matrix(columns=None)
g1_ldaDFV = g1_ldaDF[:10][['topic_%d' %x for x in range(10)]]
g1_ldaDFVisN = g1_ldaDF[:10][['url']]
g1_ldaDFVis = g1_ldaDFV.as_matrix(columns=None)

# extract 'title' part from each url
for i in range(10):
    g0_name = re.search('2016/.*/.*/(.*)', g0_ldaDFVisN.url.iloc[i]).group(1)
    g0_ldaDFVisN.url.iloc[i] = g0_name
    g1_name = re.search('2016/.*/.*/(.*)', g1_ldaDFVisN.url.iloc[i]).group(1)
    g1_ldaDFVisN.url.iloc[i] = g1_name
g0_ldaDFVisNames = g0_ldaDFVisN.as_matrix(columns=None)
g1_ldaDFVisNames = g1_ldaDFVisN.as_matrix(columns=None)

# stacked bar: dailymail0
N = 10
ind = np.arange(N)
K = d0_lda.num_topics  # N documents, K topics
ind = np.arange(N)  # the x-axis locations for the novels
width = 0.5  # the width of the bars
plots = []
height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, d0_ldaDFVis[:, k], width, color=color)
    else:
        p = plt.bar(ind, d0_ldaDFVis[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += d0_ldaDFVis[:, k]
    plots.append(p)

plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
plt.ylabel('Topics')
plt.title('Topics in Press Releases')
plt.xticks(ind+width/2, d0_ldaDFVisNames, rotation='vertical')
plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc='center left', frameon=True,  bbox_to_anchor = (1, .5))
plt.show()
plt.close()

'''
--------------------------------------------------------------------------------
Word embedding (from CCA Notebook 4)
--------------------------------------------------------------------------------
In this part, with word embedding technique, I examine the use of specific
words in order to see how each newspaper resemble or differ from each other.
More specifically, this is to analyze the converging/divering views of the
two newspapers on the impact of automation (and related matters) and
immigration (and related matters) on economy and job market.
--------------------------------------------------------------------------------
'''
'''Word2Vec'''
# initialize stemmer and stop words
stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()

# tokenize and normalize text: dailymail
dailymailDF['tokenized_sents'] = dailymailDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
dailymailDF['normalized_sents'] = dailymailDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])
# tokenize and normalize text: guardian
guardianDF['tokenized_sents'] = guardianDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
guardianDF['normalized_sents'] = guardianDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

# get word vectors
# dailymailW2V = gensim.models.word2vec.Word2Vec(dailymailDF['normalized_sents'].sum())
# guardianW2V = gensim.models.word2vec.Word2Vec(guardianDF['normalized_sents'].sum())
# save trained models
# dailymailW2V.save('model/dailymailW2V.model')
# guardianW2V.save('model/guardianW2V.model')
# load trained models
dailymailW2V = gensim.models.word2vec.Word2Vec.load('model/dailymailW2V.model')
guardianW2V = gensim.models.word2vec.Word2Vec.load('model/guardianW2V.model')

# examine word vectors
dailymailW2V.most_similar('ai', topn=20) # similar vectors
guardianW2V.most_similar('trade', topn=20) # similar vectors

dailymailW2V.most_similar(positive=['international', 'trade']) # semantic equation
guardianW2V.most_similar(positive=['international', 'trade']) # semantic equation

dailymailW2V.doesnt_match(['invest', 'investment', 'business', 'learning']) # least match
guardianW2V.doesnt_match(['invest', 'investment', 'business', 'learning']) # least match


'''Doc2Vec'''
# add tags
keywords = ['ai', 'automat', 'robot', 'globalis', 'immigr', 'trade', 'econom', 'job', 'labour', 'growth', 'declin', 'opportun', 'risk']

d_taggedDocs = []
for index, row in dailymailDF.iterrows():
    #Just doing a simple keyword assignment
    docKeywords = [s for s in keywords if s in row['normalized_tokens']]
    docKeywords.append(row['url'])
    docKeywords.append(row['date']) #This lets us extract individual documnets since doi's are unique
    d_taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words = row['normalized_tokens'], tags = docKeywords))
dailymailDF['TaggedArticles'] = d_taggedDocs

g_taggedDocs = []
for index, row in guardianDF.iterrows():
    #Just doing a simple keyword assignment
    docKeywords = [s for s in keywords if s in row['normalized_tokens']]
    docKeywords.append(row['url'])
    docKeywords.append(row['date']) #This lets us extract individual documnets since doi's are unique
    g_taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words = row['normalized_tokens'], tags = docKeywords))
guardianDF['TaggedArticles'] = g_taggedDocs

# train
# dailymailD2V = gensim.models.doc2vec.Doc2Vec(dailymailDF['TaggedArticles'], size = 100) #Limiting to 100 dimensions
# guardianD2V = gensim.models.doc2vec.Doc2Vec(guardianDF['TaggedArticles'], size = 100) #Limiting to 100 dimensions
# save trained models
# dailymailD2V.save('model/dailymailD2V.model')
# guardianD2V.save('model/guardianD2V.model')
# load trained models
dailymailD2V = gensim.models.doc2vec.Doc2Vec.load('model/dailymailD2V.model')
guardianD2V = gensim.models.doc2vec.Doc2Vec.load('model/guardianD2V.model')

# heatmap
d_heatmapMatrix = []
g_heatmapMatrix = []
for tagOuter in keywords:
    d_column = []
    g_column = []
    d_tagVec = dailymailD2V.docvecs[tagOuter].reshape(1, -1)
    g_tagVec = guardianD2V.docvecs[tagOuter].reshape(1, -1)
    for tagInner in keywords:
        d_column.append(sklearn.metrics.pairwise.cosine_similarity(d_tagVec, dailymailD2V.docvecs[tagInner].reshape(1, -1))[0][0])
        g_column.append(sklearn.metrics.pairwise.cosine_similarity(g_tagVec, guardianD2V.docvecs[tagInner].reshape(1, -1))[0][0])
    d_heatmapMatrix.append(d_column)
    g_heatmapMatrix.append(g_column)
d_heatmapMatrix = np.array(d_heatmapMatrix)
g_heatmapMatrix = np.array(g_heatmapMatrix)
d_heatmapMatrix.shape

# word-to-word: dailymail
fig, ax = plt.subplots()
hmap = ax.pcolor(d_heatmapMatrix, cmap='terrain')
cbar = plt.colorbar(hmap)
cbar.set_label('cosine similarity', rotation=270)
a = ax.set_xticks(np.arange(d_heatmapMatrix.shape[1]) + 0.5, minor=False)
a = ax.set_yticks(np.arange(d_heatmapMatrix.shape[0]) + 0.5, minor=False)
a = ax.set_xticklabels(keywords, minor=False, rotation=270)
a = ax.set_yticklabels(keywords, minor=False)
plt.savefig('image/figW2WD', bbox_inches = 'tight')
plt.show()
plt.close()

# word-to-word: guardian
fig, ax = plt.subplots()
hmap = ax.pcolor(g_heatmapMatrix, cmap='terrain')
cbar = plt.colorbar(hmap)
cbar.set_label('cosine similarity', rotation=270)
a = ax.set_xticks(np.arange(g_heatmapMatrix.shape[1]) + 0.5, minor=False)
a = ax.set_yticks(np.arange(g_heatmapMatrix.shape[0]) + 0.5, minor=False)
a = ax.set_xticklabels(keywords, minor=False, rotation=270)
a = ax.set_yticklabels(keywords, minor=False)
plt.savefig('image/figW2WG', bbox_inches = 'tight')
plt.show()
plt.close()

'''Projection'''
from functions import normalize, dimension, makeDF, Coloring, PlotDimension

#words to create dimensions
TargetWords = ['good', 'better', 'positive', 'opportunity', 'bad', 'worse', 'negative', 'risk', 'decline']
TargetWords += ['business', 'corporation', 'corporate', 'labour', 'worker', 'working']
TargetWords += ['economic', 'economy', 'social', 'society']

#words we will be mapping
TargetWords += ['ai', 'automation', 'automated', 'robot', 'globalisation', 'immigrant', 'immigration', 'trade']

# get submatrix
d_wordsSubMatrix = []
g_wordsSubMatrix = []
for word in TargetWords:
    d_wordsSubMatrix.append(dailymailW2V[word])
    g_wordsSubMatrix.append(guardianW2V[word])
d_wordsSubMatrix = np.array(d_wordsSubMatrix)
g_wordsSubMatrix = np.array(g_wordsSubMatrix)

d_pcaWords = sklearn.decomposition.PCA(n_components = 50).fit(d_wordsSubMatrix)
g_pcaWords = sklearn.decomposition.PCA(n_components = 50).fit(g_wordsSubMatrix)
d_reducedPCA = d_pcaWords.transform(d_wordsSubMatrix)
g_reducedPCA = g_pcaWords.transform(g_wordsSubMatrix)

# #T-SNE is theoretically better, but you should experiment
d_tsneWords = sklearn.manifold.TSNE(n_components = 2).fit_transform(d_reducedPCA)
g_tsneWords = sklearn.manifold.TSNE(n_components = 2).fit_transform(g_reducedPCA)

# plot the keywords
fig = plt.figure() #figsize = (10,6)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.plot(d_tsneWords[:, 0], d_tsneWords[:, 1], alpha = 0) #Making the points invisible
for i, word in enumerate(TargetWords):
    ax.annotate(word, (d_tsneWords[:, 0][i],d_tsneWords[:, 1][i]), size =  20 * (len(TargetWords) - i) / len(TargetWords))
plt.xticks(())
plt.yticks(())
plt.show()
plt.close()

fig = plt.figure() #figsize = (10,6)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.plot(g_tsneWords[:, 0], g_tsneWords[:, 1], alpha = 0) #Making the points invisible
for i, word in enumerate(TargetWords):
    ax.annotate(word, (g_tsneWords[:, 0][i],g_tsneWords[:, 1][i]), size =  20 * (len(TargetWords) - i) / len(TargetWords))
plt.xticks(())
plt.yticks(())
plt.show()
plt.close()

# define dimensions
d_Dimension1 = dimension(dailymailW2V, ['good', 'better', 'positive', 'opportunity', 'growth'], ['bad', 'worse', 'negative', 'risk', 'decline'])
d_Dimension2 = dimension(dailymailW2V, ['business', 'corporation', 'corporate'], ['labour', 'worker', 'working'])
d_Dimension3 = dimension(dailymailW2V, ['economic', 'economy'], ['social', 'society'])
g_Dimension1 = dimension(guardianW2V, ['good', 'better', 'positive', 'opportunity', 'growth'], ['bad', 'worse', 'negative', 'risk', 'decline'])
g_Dimension2 = dimension(guardianW2V, ['business', 'corporation', 'corporate'], ['labour', 'worker', 'working'])
g_Dimension3 = dimension(guardianW2V, ['economic', 'economy'], ['social', 'society'])

# define words to be projected
Keywords = ['ai', 'automation', 'automated', 'robot', 'globalisation', 'immigrant', 'immigration', 'trade']
# get dimensions
d_DimensionDF = makeDF(dailymailW2V, Keywords, d_Dimension1, d_Dimension2, d_Dimension3)
g_DimensionDF = makeDF(guardianW2V, Keywords, g_Dimension1, g_Dimension2, d_Dimension3)

# projecting: dailymail
fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(131)
PlotDimension(ax1, d_DimensionDF, 'Dimension1')
ax2 = fig.add_subplot(132)
PlotDimension(ax2, d_DimensionDF, 'Dimension2')
ax3 = fig.add_subplot(133)
PlotDimension(ax3, d_DimensionDF, 'Dimension3')
plt.savefig('image/figProjectionD')
plt.show()
plt.close()

# projecting: guardian
fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(131)
PlotDimension(ax1, g_DimensionDF, 'Dimension1')
ax2 = fig.add_subplot(132)
PlotDimension(ax2, g_DimensionDF, 'Dimension2')
ax3 = fig.add_subplot(133)
PlotDimension(ax3, g_DimensionDF, 'Dimension3')
plt.savefig('image/figProjectionG')
plt.show()
plt.close()
