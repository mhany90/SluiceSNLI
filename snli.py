from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import numpy as np
import codecs
np.random.seed(1337)  # for reproducibility

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, TimeDistributed, Reshape,  Bidirectional, Concatenate, GaussianNoise
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal

def read_embeds(efile, word_index):    
    embedding_index = {}
    f = open(efile, "r")
    for line in f:
        values = line.split(' ')
        embedding_dim = len(values)-1
        word = values[0].strip()
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, embedding_dim

def get_data(filename):
  file = codecs.open(filename)
  first,second,Y = [], [], []
  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  for line in file.readlines()[1:]:
      separated = line.split('\t')
      if separated[0] == '-':
          continue
      else:
          first.append(separated[1].strip())
          second.append(separated[2].strip())
          Y.append(LABELS[separated[0]])

  Y = np_utils.to_categorical(Y, len(LABELS))
  return first, second, Y

"""
def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    print(label, s1, s2)
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  print(max(len(x.split()) for x in left))
  print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y
"""


def make_model(word_index, max_seq, embeds, embed_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, LAYERS):

    embedding_layer = Embedding(len(word_index) + 1,
                             100,
#                            weights = [embeds],
                            input_length=max_seq,
                            trainable = True)

    premise_input = Input(shape=(max_seq,), dtype='int32')
    hypo_input = Input(shape=(max_seq,), dtype='int32')

    translate = TimeDistributed(Dense(SENT_REP_SIZE, activation=ACTIVATION))

    
    prem_embedded_sequences = embedding_layer(premise_input)
    hypo_embedded_sequences = embedding_layer(hypo_input)
    
    #translation layer for fixed embeddings
    prem = translate(prem_embedded_sequences)
    hypo = translate(hypo_embedded_sequences)
   
    #stack?
    #for l in range(LAYERS - 1):
    BiLSTM = Bidirectional(LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP))
    prem = BiLSTM(prem)
    hypo = BiLSTM(hypo)
    prem = BatchNormalization()(prem)
    hypo = BatchNormalization()(hypo)

    merged = keras.layers.concatenate([prem, hypo], axis = -1)
    merged = Dropout(DP)(merged)

    #final dense layers
    for i in range(3):
        merged = Dense(4 * SENT_REP_SIZE, activation = ACTIVATION)(merged)
        merged = Dropout(DP)(merged)
        merged = BatchNormalization()(merged)

    pred = Dense(3, activation='softmax')(merged)
    
    return premise_input, hypo_input, merged, pred
    
#load from file
training = get_data('/data/s3094723/thesis/snli/snli_1.0/train.sem.tag.sent')
validation = get_data('/data/s3094723/thesis/snli/snli_1.0/dev.sem.tag.sent')
test = get_data('/data/s3094723/thesis/snli/snli_1.0/test.sem.tag.sent')

#training = get_data('/data/s3094723/thesis/snli/snli_1.0/snli_1.0_train.jsonl')
#validation = get_data('/data/s3094723/thesis/snli/snli_1.0/snli_1.0_dev.jsonl')
#test = get_data('/data/s3094723/thesis/snli/snli_1.0/snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(training[0] + training[1])

#set variables
VOCAB_LEN = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
TRAIN_EMBED = False
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 42
MAX_LEN = 42
DP = 0.2
OPTIMIZER = 'rmsprop'
ACTIVATION = 'relu'
SENT_REP_SIZE = 100
EFILE = "/data/s3094723/embeddings/en/glove.840B.300d.txt"
L2 = 4e-6
LAYERS = 3

#load data
to_sequences = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
convert_data = lambda data: (to_sequences(data[0]), to_sequences(data[1]), data[2])

training = convert_data(training)
validation = convert_data(validation)
test = convert_data(test)
print(training[0].shape, training[1].shape, training[2].shape)
word_index = tokenizer.word_index

print('Build model...')
print('Vocab size =', VOCAB_LEN)


#load embeddings
embedding_matrix, embedding_dim = read_embeds(EFILE, word_index)
print('Total number of null word embeddings:')
print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

#build
#oder: word_index, max_seq, embeds, embed_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, no os layers

premise_input, hypo_input, final, preds = make_model(word_index, MAX_LEN, embedding_matrix, embedding_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, LAYERS)
fmodel = Model(inputs=[premise_input, hypo_input], outputs=preds)
fmodel.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
fmodel.summary()

callbacks = [EarlyStopping(patience=PATIENCE)]
fmodel.fit([training[0],training[1]] , training[2], 
         batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, 
         validation_data=([validation[0], validation[1]], validation[2]), callbacks = callbacks)


loss, acc = fmodel.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
