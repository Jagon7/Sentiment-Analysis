############################################################## Project 2: Sentiment Analysis ###########################################################################
#################################### Part I: Bloomberg Inflation Analysis #######################################
from newsapi import NewsApiClient
import pandas as pd
import numpy as np
import csv

api = NewsApiClient(api_key='PUT YOUR KEY HERE')
all_articles = api.get_everything(qintitle='inflation', sources='bloomberg', from_param='2022-11-29', to='2022-12-28', language='en')
all_articles = all_articles['articles']

# Filter away Podcast and Radio & Save as csv file
filter_criteria = ['/audio/']
filtered_allArticles = []
for i in range(len(all_articles)):
    blocked = False
    for item in filter_criteria:
        if item in all_articles[i]['url']:
            blocked = True
            break
    if blocked == False:
        filtered_allArticles.append(all_articles[i])

filtered_allArticles = pd.DataFrame (filtered_allArticles, columns = ['source', 'author', 'title', 'discription', 'url', 'urlToImage', 'publishedAt', 'content'])
filtered_allArticles.to_csv('filtered_allArticles.csv', index=False, header=True)

# Sentiment Analysis
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
import pickle
import nltk
import logging
import multiprocessing
from datetime import datetime
from re import sub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from time import time 
from unidecode import unidecode
from gensim.models import Word2Vec
from collections import defaultdict
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.models.phrases import Phrases, Phraser
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet
import spacy 
import stanza 
import spacy_stanza
from negspacy.negation import Negex
from negspacy.termsets import termset 

filtered_allArticles = pd.read_csv('filtered_allArticles.csv')

# Clean the text
sep = ' … '
for i in range(len(filtered_allArticles)):
    filtered_allArticles['content'][i] = filtered_allArticles['content'][i].split(sep, 1)[0]

# Define a function to clean the text
def text_to_word_list(text):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)

    return text

# Cleaning the text in the column
filtered_allArticles['Cleaned_content'] = filtered_allArticles['content'].apply(text_to_word_list)

# Tokenization, Stopwords Removal, POS Tagging
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

filtered_allArticles['POS tagged'] = filtered_allArticles['Cleaned_content'].apply(token_stop_pos)

# Obtaining the stem words – Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

filtered_allArticles['Lemma'] = filtered_allArticles['POS tagged'].apply(lemmatize)



# TextBlob
fin_data = pd.DataFrame(filtered_allArticles[['content', 'Lemma']])

from textblob import TextBlob

# function to calculate subjectivity
def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity

# function to calculate polarity
def getPolarity(review):
    return TextBlob(review).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

fin_data['Subjectivity'] = fin_data['Lemma'].apply(getSubjectivity) 
fin_data['Polarity'] = fin_data['Lemma'].apply(getPolarity) 
fin_data['Analysis'] = fin_data['Polarity'].apply(analysis)

tb_counts = fin_data.Analysis.value_counts()
plt.figure(figsize=(10, 7))
plt.pie(tb_counts.values, labels = tb_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
plt.legend();

## VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# function to calculate vader sentiment
def vadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']
fin_data['Vader Sentiment'] = fin_data['Lemma'].apply(vadersentimentanalysis)

# function to analyse
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'
        
fin_data['Vader Analysis'] = fin_data['Vader Sentiment'].apply(vader_analysis)

vader_counts = fin_data['Vader Analysis'].value_counts()
plt.figure(figsize=(10, 7))
plt.pie(vader_counts.values, labels = vader_counts.index, explode = (0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.legend();

# SentiWordNet
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn

def sentiwordnetanalysis(pos_data):
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
        # print(swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score())
    if not tokens_count:
        return 0
    if sentiment>0:
        return "Positive"
    if sentiment==0:
        return "Neutral"
    else:
        return "Negative"

fin_data['SWN analysis'] = filtered_allArticles['POS tagged'].apply(sentiwordnetanalysis)
swn_counts= fin_data['SWN analysis'].value_counts()
plt.figure(figsize=(10, 7))
plt.pie(swn_counts.values, labels = swn_counts.index, explode = (0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.legend();

# Selection
for i in range(len(fin_data)):
    if fin_data['Analysis'][i] == 'Positive':
        fin_data['Analysis'][i] = 1
    elif fin_data['Analysis'][i] == 'Negative':
        fin_data['Analysis'][i] = -1
    else:
        fin_data['Analysis'][i] = 0

    if fin_data['Vader Analysis'][i] == 'Positive':
        fin_data['Vader Analysis'][i] = 1
    elif fin_data['Vader Analysis'][i] == 'Negative':
        fin_data['Vader Analysis'][i] = -1
    else:
        fin_data['Vader Analysis'][i] = 0

    if fin_data['SWN analysis'][i] == 'Positive':
        fin_data['SWN analysis'][i] = 1
    elif fin_data['SWN analysis'][i] == 'Negative':
        fin_data['SWN analysis'][i] = -1
    else:
        fin_data['SWN analysis'][i] = 0

fin_data['finalSentiment'] = [0] * len(fin_data)
for i in range(len(fin_data)):
    fin_data['finalSentiment'][i] = fin_data['Analysis'][i] + fin_data['Vader Analysis'][i] + fin_data['SWN analysis'][i]

fin_data['Res'] = fin_data['finalSentiment']
for i in range(len(fin_data)):
    if fin_data['finalSentiment'][i] == 3:
        fin_data['Res'][i] = 'Strong Positive'
    elif fin_data['finalSentiment'][i] == 2:
        fin_data['Res'][i] = 'Positive'
    elif -1 <= fin_data['finalSentiment'][i] <= 1:
        fin_data['Res'][i] = 'Neutral'
    elif fin_data['finalSentiment'][i] == -2:
        fin_data['Res'][i] = 'Negative'
    elif fin_data['finalSentiment'][i] == -3:
        fin_data['Res'][i] = 'Stong Negative'

fin_data.to_csv('fin_data.csv', index=False, header=True)

# Visualization
# 1. Sentiment Scores for Inflation News
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

plt.close('all')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.2})
fig.set_figheight(10)
fig.set_figwidth(10)
ax = plt.subplot(111)    # The big subplot
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.bar(fin_data[fin_data['Subjectivity'] > 0]['Subjectivity'].index, fin_data[fin_data['Subjectivity'] > 0]['Subjectivity'], width = 0.5, color='r')
ax1.bar(fin_data[fin_data['Subjectivity'] < 0]['Subjectivity'].index, fin_data[fin_data['Subjectivity'] < 0]['Subjectivity'], width=0.5, color='b')
ax2.bar(fin_data[fin_data['Polarity'] > 0]['Polarity'].index, fin_data[fin_data['Polarity'] > 0]['Polarity'], width = 0.5, color='r')
ax2.bar(fin_data[fin_data['Polarity'] < 0]['Polarity'].index, fin_data[fin_data['Polarity'] < 0]['Polarity'], width = 0.5, color='b')
ax3.bar(fin_data[fin_data['Vader Sentiment'] > 0]['Vader Sentiment'].index, fin_data[fin_data['Vader Sentiment'] > 0]['Vader Sentiment'], width=0.5, color='r')
ax3.bar(fin_data[fin_data['Vader Sentiment'] < 0]['Vader Sentiment'].index, fin_data[fin_data['Vader Sentiment'] < 0]['Vader Sentiment'], width=0.5, color='b')

# Set common labels
ax.set_xlabel('News')
ax.set_ylabel('Score')

ax1.set_title('Subjectivity')
ax2.set_title('Polarity')
ax3.set_title('Vader Sentiment')

plt.savefig('Sentiment Scores for Inflation News.png', dpi=300);


# 2. Time series chart of rolling 5 period mean of Vader Sentiment
vader_rolling_df = pd.DataFrame({'Vader Sentiment': fin_data['Vader Sentiment'], 'PublishedAt': filtered_allArticles['publishedAt']})

new_format = "%Y-%m-%d"
for i in range(len(vader_rolling_df)):
    vader_rolling_df['PublishedAt'][i] = datetime.strptime(vader_rolling_df['PublishedAt'][i], "%Y-%m-%dT%H:%M:%SZ")  # "%Y-%m-%dT%H:%M:%SZ"
    vader_rolling_df['PublishedAt'][i] = vader_rolling_df['PublishedAt'][i].strftime(new_format)

vader_rolling_df = vader_rolling_df.sort_values('PublishedAt').reset_index(drop=True)

vader_rolling_group = vader_rolling_df.groupby('PublishedAt')['Vader Sentiment'].sum().rolling(5).mean()
vader_rolling_group_df = pd.DataFrame({'PublishedAt':vader_rolling_group.index, 'Rolling Mean Score':vader_rolling_group.values})

vader_rolling_plot = vader_rolling_df.groupby('PublishedAt')['Vader Sentiment'].sum()
vader_rolling_plot_df = pd.DataFrame({'PublishedAt':vader_rolling_plot.index, 'Daily Score':vader_rolling_plot.values})

fig, ax = plt.subplots(1, 1, sharex=True, gridspec_kw={'hspace': 0})
fig.set_figheight(5)
fig.set_figwidth(15)
ax = plt.subplot(111)
ax.plot(vader_rolling_group_df['PublishedAt'], vader_rolling_group_df['Rolling Mean Score'], linewidth=3, color='black')
ax.bar(vader_rolling_plot_df[vader_rolling_plot_df['Daily Score'] > 0]['PublishedAt'], vader_rolling_plot_df[vader_rolling_plot_df['Daily Score'] > 0]['Daily Score'], width=0.5, color='r')
ax.bar(vader_rolling_plot_df[vader_rolling_plot_df['Daily Score'] < 0]['PublishedAt'], vader_rolling_plot_df[vader_rolling_plot_df['Daily Score'] < 0]['Daily Score'], width = 0.5, color='b')

# Set common labels
ax.set_xlabel('Published Date')
ax.set_ylabel('Rolling 5-period Vader Sentiment Mean Score')
ax.set_title('Time series chart of rolling 5 period mean of Vader Sentiment')
plt.xticks(rotation=60)
fig.set_tight_layout(True)
plt.savefig('Time series chart of rolling 5 period mean of Vader Sentiment.png', dpi=300);


#################################### Part II: Twitter Analysis #######################################
