from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import numpy as np
import codecs
from collections import defaultdict
np.random.seed(1337)  # for reproducibility

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, TimeDistributed, Reshape,  Bidirectional, Concatenate, GaussianNoise, Lambda, Subtract, Multiply
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
from max import  TemporalMaxPooling

from shared import WeightedCombinationLayer


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

#get tag2id 
def t2i(filenames):
    tag_to_id = defaultdict(lambda: len(tag_to_id))
    for file in filenames:
        file = codecs.open(file)
        for line in file.readlines()[12:]:
            if line != '\n':
                separated = line.split()
                tag_to_id[separated[1].strip()]

    return tag_to_id

def get_tag_data(filename, MAX_LEN, tag_to_id):
  file = codecs.open(filename)
  sent_X, sent_Y, curr_words, curr_tags = [],[],[],[]

  for line in file.readlines()[12:]:
      if line != '\n':
          separated = line.split()
          curr_words.append(separated[0].strip())
          curr_tags.append(tag_to_id[separated[1].strip()])
      else:
          sent_Y.append(curr_tags)
          sent_X.append(curr_words)
          curr_words, curr_tags = [], []
  sent_Y = pad_sequences(sent_Y,  MAX_LEN)
  print("len(tag_to_id): ", len(tag_to_id))
  Y_tag = np_utils.to_categorical(sent_Y, len(tag_to_id))
  print("Y_TAG: ", Y_tag.shape)
  return Y_tag


def make_model(word_index, max_seq, embeds, embed_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, LAYERS, NO_TAGS):

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = TRAIN_EMBED)

    premise_input = Input(shape=(max_seq,), dtype='int32')
    hypo_input = Input(shape=(max_seq,), dtype='int32')

    translate = TimeDistributed(Dense(int(SENT_REP_SIZE * 2), activation=ACTIVATION))
    
    prem_embedded_sequences = embedding_layer(premise_input)
    hypo_embedded_sequences = embedding_layer(hypo_input)
    
   #translation layer for fixed embeddings
    prem = translate(prem_embedded_sequences)
    hypo = translate(hypo_embedded_sequences)

    #prem = prem_embedded_sequences 
    #hypo = hypo_embedded_sequences

    #auxillary 
    #prem_out_tags = TimeDistributed(Dense(NO_TAGS, activation='softmax'))(prem) 
    #hypo_out_tags = TimeDistributed(Dense(NO_TAGS, activation='softmax'))(hypo)
  
    #snli
    BiLSTM_snli_1 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)
    BiLSTM_snli_2 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)
    #sem
    BiLSTM_tag_11 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)
    BiLSTM_tag_12 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)
    BiLSTM_tag_21 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)
    BiLSTM_tag_22 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)


    prem_snli_1 = BiLSTM_snli_1(prem)
    prem_snli_2 = BiLSTM_snli_2(prem)

    prem_tag_1 = BiLSTM_tag_11(prem)
    prem_tag_2 = BiLSTM_tag_12(prem)

    hypo_snli_1 = BiLSTM_snli_1(hypo)
    hypo_snli_2 = BiLSTM_snli_2(hypo)

    hypo_tag_1 = BiLSTM_tag_21(hypo)  
    hypo_tag_2 = BiLSTM_tag_22(hypo)

    #alphas
    alpha_snli_prem = WeightedCombinationLayer()([prem_snli_1,prem_snli_2, prem_tag_1, prem_tag_2])
    alpha_tag_prem = WeightedCombinationLayer()([prem_snli_1,prem_snli_2, prem_tag_1, prem_tag_2])

    alpha_snli_hypo = WeightedCombinationLayer()([hypo_snli_1, hypo_snli_2, hypo_tag_1,  hypo_tag_2])
    alpha_tag_hypo = WeightedCombinationLayer()([hypo_snli_1, hypo_snli_2, hypo_tag_1,  hypo_tag_2])


    #auxillary 
    prem_out_tags = TimeDistributed(Dense(NO_TAGS, activation='softmax'))(alpha_tag_prem)
    hypo_out_tags = TimeDistributed(Dense(NO_TAGS, activation='softmax'))(alpha_tag_hypo)

    #shortcut
    shortcuted_prem = keras.layers.concatenate([alpha_snli_prem, prem_out_tags, prem], axis = -1)
    shortcuted_hypo = keras.layers.concatenate([alpha_snli_hypo, hypo_out_tags, hypo], axis = -1)
        
    #level 2
    BiLSTM = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP, return_sequences = True)
    prem = BiLSTM(shortcuted_prem)
    hypo = BiLSTM(shortcuted_hypo)

    print("prem:", prem.shape)
    prem = TemporalMaxPooling()(prem)
    hypo = TemporalMaxPooling()(hypo)

    #prem = BatchNormalization()(prem)
    #hypo = BatchNormalization()(hypo)

    sub = Subtract()([prem, hypo])
    mult = Multiply()([prem, hypo])

    merged = keras.layers.concatenate([sub, mult], axis = -1)
    merged = Dropout(DP)(merged)

    #final dense layers
    for i in range(2):
        merged = Dense(2 * SENT_REP_SIZE, activation = ACTIVATION)(merged)
        merged = Dropout(DP)(merged)
  #      merged = BatchNormalization()(merged)

    pred = Dense(3, activation='softmax')(merged)
    
    return premise_input, hypo_input, merged, pred, prem_out_tags, hypo_out_tags 
  

MAX_LEN = 42
  
#load from file
training = get_data('/data/s3094723/thesis/snli/snli_1.0/train.formated')
validation = get_data('/data/s3094723/thesis/snli/snli_1.0/dev.formated')
test = get_data('/data/s3094723/thesis/snli/snli_1.0/test.formated')

#training = get_data('/data/s3094723/thesis/snli/snli_1.0/snli_1.0_train.jsonl')
#validation = get_data('/data/s3094723/thesis/snli/snli_1.0/snli_1.0_dev.jsonl')
#test = get_data('/data/s3094723/thesis/snli/snli_1.0/snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(training[0] + training[1])

#tag data
#get tag2id 
fnames = ['/data/s3094723/thesis/snli/snli_1.0/train.formated.1.pos', '/data/s3094723/thesis/snli/snli_1.0/train.formated.2.pos',
'/data/s3094723/thesis/snli/snli_1.0/dev.formated.1.pos', '/data/s3094723/thesis/snli/snli_1.0/dev.formated.2.pos',
'/data/s3094723/thesis/snli/snli_1.0/test.formated.1.pos','/data/s3094723/thesis/snli/snli_1.0/test.formated.2.pos']

tag_to_id = t2i(fnames)

training_tag1 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/train.formated.1.pos', MAX_LEN, tag_to_id)
training_tag2 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/train.formated.2.pos', MAX_LEN, tag_to_id)

validation_tag1 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/dev.formated.1.pos', MAX_LEN, tag_to_id)
validation_tag2 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/dev.formated.2.pos', MAX_LEN, tag_to_id)

test_tag1 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/test.formated.1.pos', MAX_LEN, tag_to_id)
test_tag2 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/test.formated.2.pos', MAX_LEN, tag_to_id)

#set variables
NO_TAGS = len(tag_to_id)
print('NO_TAGS: ', NO_TAGS)
VOCAB_LEN = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
TRAIN_EMBED = False
BATCH_SIZE = 512
PATIENCE = 2 # 8
MAX_EPOCHS = 42
DP = 0.2
OPTIMIZER = 'adam'
ACTIVATION = 'relu'
SENT_REP_SIZE = 300
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

premise_input, hypo_input, final, preds,  prem_out_tags, hypo_out_tags = make_model(word_index, MAX_LEN, embedding_matrix, 
                                                                                    embedding_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, LAYERS, NO_TAGS)
fmodel = Model(inputs=[premise_input, hypo_input], outputs=[preds,  prem_out_tags, hypo_out_tags])
fmodel.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=[1., 0.05, 0.05])
fmodel.summary()

callbacks = [EarlyStopping(patience=PATIENCE, monitor='val_dense_6_acc')]
fmodel.fit([training[0],training[1]] , [training[2], training_tag1, training_tag2], 
         batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, 
         validation_data=([validation[0], validation[1]], [validation[2], validation_tag1, validation_tag2]), callbacks = callbacks)


#loss, acc = fmodel.evaluate([test[0], test[1]], [test[2], test_tag1, test_tag2], batch_size=BATCH_SIZE)
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
all  = fmodel.evaluate([test[0], test[1]], [test[2], test_tag1, test_tag2], batch_size=BATCH_SIZE)

print(all)
