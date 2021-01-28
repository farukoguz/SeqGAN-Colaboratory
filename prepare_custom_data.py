#!/usr/bin/env python
# coding: utf-8

# Prepare training data for seq-gan
# - Corpora: http://www.nltk.org/howto/corpus.html

import json
import numpy as np
import pandas as pd
import re
import string

from collections import Counter
from nltk.tokenize import WhitespaceTokenizer 

# Params
MAX_SENTENCE_LENGTH = 16
VOCAB_SIZE = 5000

TEXT_DATA_PATH = 'data/movie.txt'
TRAIN_DATA_PATH = 'data/gan_train.txt'
VOCAB_PATH = 'data/vocabulary.json'

# Download raw text dataset 
# The example here is a movie review dataset from nltk: from nltk.corpus import movie_reviews)
# !curl https://raw.githubusercontent.com/davidrossouw/data/main/nlp/movie_data.txt --output data/movie.txt

# Load raw dataset
df = pd.read_csv(TEXT_DATA_PATH)

TRAINING_EXAMPLES = len(df)

tokenizer = WhitespaceTokenizer() 
regex = re.compile('[%s]' % re.escape(string.punctuation))

def preprocess(sentence, max_length): 
    tokens = tokenizer.tokenize(regex.sub('', sentence).lower())
    # Pad to max_length
    if len(tokens) < max_length:
        tokens.extend([0] * (MAX_SENTENCE_LENGTH - len(tokens)))
    # Crop to max_length
    elif len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokens

# Apply preprocessing
df['tokens'] = df['review'].apply(lambda s: preprocess(s, MAX_SENTENCE_LENGTH))

# Create vocabulary, id of 0 = OOV. 10k most common
all_tokens = [t for sent in df['tokens'].tolist() for t in sent]
counted = Counter(all_tokens).most_common(4999)
vocabulary = {w[0]:i+1 for i, w in enumerate(counted)}
vocabulary['<unk>'] = 0
inverse_vocabulary = {v:k for k,v in vocabulary.items()}

# Convert to numpy array
df['vector'] = df['tokens'].apply(lambda x: [vocabulary.get(w, 0) for w in x])
train_set = np.array(df['vector'].tolist())
train_set.shape

# Save training set
np.savetxt(TRAIN_DATA_PATH, train_set, fmt='%d')

# Save vocabulary
with open(VOCAB_PATH, 'w') as f:
    json.dump(vocabulary, f)




