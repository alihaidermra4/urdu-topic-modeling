# Execute First
from typing import List, Any, Generator

import gensim
from gensim.utils import simple_preprocess
from gensim.utils import tokenize
from gensim.parsing import stem_text
from gensim import corpora
from gensim import models
from gensim.models import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary
import pyLDAvis
import pickle
import pyLDAvis.gensim_models
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')

stopwords = pd.read_excel('Stopwords.xlsx')
stopwords.dropna(subset=['Stopwords'], inplace=True)
print(stopwords.head())

urdu_punctuation = '؛‘’“""”«»،؟.ء'
all_punctuation = string.punctuation + urdu_punctuation

stopwords['Stopwords'] = [re.sub(f'[{re.escape(all_punctuation)}]', '', str(stopword)) for stopword in stopwords['Stopwords']]
print(stopwords[0:])

df = pd.read_excel('Headlines.xlsx')
print(df.head())

data_pre = df.drop(columns=['Labels'])
data_pre.dropna(subset=['Sentences'], inplace=True)
data_pre['Sentences'] = [re.sub(f'[{re.escape(all_punctuation)}]', '', str(sentence)) for sentence in data_pre['Sentences']]
print(data_pre[0:])

# PreProcessing

def remove_stopwords(texts, stw):
    result = []

    for doc in texts:
        words = word_tokenize(doc)
        filtered_doc = [word for word in words if word not in stw]
        result.append(filtered_doc)
    return result

data = data_pre['Sentences'].values.tolist()
print(data[0:])
print((len(data[0:])))

stop_data = stopwords['Stopwords'].values.tolist()
print((stop_data[0:]))
print(len((stop_data[0:])))

data_words = remove_stopwords(data, stop_data)

print(data_words[0:])


def sent_to_words(sentences):
    for sentence in sentences:
        # True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def removal(wordss):
    for sentence in wordss:
        # Tokenization
        tokens = tokenize(sentence, 'ur')

        # Diacritics removal
        tokens = [remove_diacritics(token) for token in tokens]

        # Space insertion errors removal
        tokens = remove_space_insertion_errors(tokens)

        # Space omission errors removal
        tokens = remove_space_omission_errors(tokens)

        # Stemming
        # You can replace this step with your preferred stemming algorithm for Urdu
        # stemmed_tokens = [stem_text(token, 1)[0] for token in tokens]

        # preprocessed_doc = ' '.join(stemmed_tokens)
        # sentence.append(preprocessed_doc)


# Create Dictionary
id2word = corpora.Dictionary(data_words)
print(id2word)
count = 0
for k, v in id2word.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
# a,b=corpus[0][0]
# print(a,b)

# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


num_topics = 5
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Visualization


perplexity = lda_model.log_perplexity(corpus)
print(f"Perplexity: {perplexity}")

coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=None, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f"Coherence Score: {coherence_lda}")
"""
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis
"""