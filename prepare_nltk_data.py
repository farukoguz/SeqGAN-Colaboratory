#!/usr/bin/env python
# coding: utf-8

# Prepare nltk training data for seq-gan
import json
import nltk
import numpy as np
import pandas as pd

from collections import Counter
from nltk.corpus import movie_reviews

# Params
MAX_SENTENCE_LENGTH = 16
VOCAB_SIZE = 5000
TRAINING_EXAMPLES = 2000

TRAIN_DATA_PATH = 'gan_train.txt'
VOCAB_PATH = 'vocabulary.json'

# Download nltk movie review dataset 
nltk.download('movie_reviews')
review_data = [(movie_reviews.words(file_id),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id)]

# Load dataset
df = pd.DataFrame(review_data, columns=['review', 'sentimnet'])
df = df[:TRAINING_EXAMPLES]
df.tail(3)

def preprocess(sentence, max_length): 
    tokens = sentence
    # Pad to max_length
    if len(tokens) < max_length:
        tokens.extend([0] * (MAX_SENTENCE_LENGTH - len(tokens)))
    # Crop to max_length
    elif len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokens

# Apply preprocessing
df['processed'] = df['review'].apply(lambda s: preprocess(s, MAX_SENTENCE_LENGTH))

# Create vocabulary, id of 0 = OOV. 10k most common
all_tokens = [t for sent in df['processed'].tolist() for t in sent]
counted = Counter(all_tokens).most_common(4999)
vocabulary = {w[0]:i+1 for i, w in enumerate(counted)}
vocabulary['<unk>'] = 0
inverse_vocabulary = {v:k for k,v in vocabulary.items()}

# Convert to numpy array
df['vector'] = df['processed'].apply(lambda x: [vocabulary.get(w, 0) for w in x])
train_set = np.array(df['vector'].tolist())
train_set.shape

# Save training set
np.savetxt(TRAIN_DATA_PATH, train_set, fmt='%d')

# Save vocabulary
with open(VOCAB_PATH, 'w') as f:
    json.dump(vocabulary, f)
