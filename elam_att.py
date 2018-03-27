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
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, TimeDistributed, Reshape,  Bidirectional, Concatenate, Multiply, Subtract, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
from max import TemporalMaxPooling
from avg import TemporalAvgPooling
from shared import WeightedCombinationLayer, WeightedCombination, WeightedCombinationLayerUnb


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

def make_model(word_index, max_seq, embeds, embed_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, LAYERS,  NO_TAGS):

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = TRAIN_EMBED)

    premise_input = Input(shape=(max_seq,), dtype='int32')
    hypo_input = Input(shape=(max_seq,), dtype='int32')

    #embed
    prem_embedded_sequences = embedding_layer(premise_input)
    hypo_embedded_sequences = embedding_layer(hypo_input)

    #tag
    BiLSTMtagp_1 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                              return_sequences = True)
    BiLSTMtagh_1 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                               return_sequences = True)

    prem_tag_1 =  BiLSTMtagp_1(prem_embedded_sequences)
    hypo_tag_1 =  BiLSTMtagh_1(hypo_embedded_sequences)

    #auxillary 
    prem_out_tags = TimeDistributed(Dense(NO_TAGS, activation='softmax'))(prem_tag_1)
    hypo_out_tags = TimeDistributed(Dense(NO_TAGS, activation='softmax'))(hypo_tag_1)

    
    #alphas 
    alpha_prem_hidden = WeightedCombinationLayerUnb()([prem_embedded_sequences, prem_tag_1])
    alpha_hypo_hidden = WeightedCombinationLayerUnb()([hypo_embedded_sequences, hypo_tag_1])

    #Encode
    BiLSTMenc_1 = Bidirectional(LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                              return_sequences = True))
    BiLSTMenc_2 = Bidirectional(LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                              return_sequences = True))
    #apply
    prem_hidden_1 = BiLSTMenc_1(prem_embedded_sequences)
    prem_hidden_2 = BiLSTMenc_2(alpha_prem_hidden)

    hypo_hidden_1 = BiLSTMenc_1(hypo_embedded_sequences)
    hypo_hidden_2 = BiLSTMenc_2(alpha_hypo_hidden)
   
    #shortcut
    alpha_prem_hidden = keras.layers.concatenate([prem_hidden_2, prem_hidden_1], axis = -1)
    alpha_hypo_hidden = keras.layers.concatenate([hypo_hidden_2, hypo_hidden_1], axis = -1)

    #attention scores for all word pairs 
    seq_prem, seq_hypo =  alpha_prem_hidden, alpha_hypo_hidden
    dot_ph = keras.layers.Dot(axes=(2, 2))([seq_hypo, seq_prem])
    #hypo scores
    scores_hypo = Lambda(lambda x: keras.activations.softmax(x))(dot_ph)
    #Transpose
    dot_ph_T = keras.layers.Permute((2, 1))(dot_ph)
    scores_prem =  Lambda(lambda x: keras.activations.softmax(x))(dot_ph_T)   

    #compute alignment and enhancements with mult + subt.
    align_prem = keras.layers.Dot((2, 1))([scores_prem, alpha_hypo_hidden])
    align_hypo = keras.layers.Dot((2, 1))([scores_hypo, alpha_prem_hidden])
  
    mult_prem = Multiply()([alpha_prem_hidden, align_prem])
    mult_hypo =	Multiply()([alpha_hypo_hidden, align_hypo])

    subt_prem =	Subtract()([alpha_prem_hidden, align_prem])
    subt_hypo = Subtract()([alpha_hypo_hidden, align_hypo])
    
    #concatenate seqs, alignment, and enhancments 
    align_prem = Dropout(DP)(Concatenate()([alpha_prem_hidden, align_prem, mult_prem, subt_prem]))
    align_hypo = Dropout(DP)(Concatenate()([alpha_hypo_hidden, align_hypo, mult_hypo, subt_hypo]))

    #projection layer to reduce complexity
    translate = TimeDistributed(Dense(SENT_REP_SIZE,activation=ACTIVATION))
   
    align_prem = translate(align_prem)
    align_hypo = translate(align_hypo)
    
    #decode
    BiLSTMdec = Bidirectional(LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                              return_sequences = True))
    final_prem = Dropout(DP)(BiLSTMdec(align_prem))
    final_hypo = Dropout(DP)(BiLSTMdec(align_hypo))
   
    average_prem = TemporalAvgPooling()(final_prem)
    average_hypo = TemporalAvgPooling()(final_hypo)

    max_prem = TemporalMaxPooling()(final_prem)
    max_hypo = TemporalMaxPooling()(final_hypo)

    merged = Concatenate()([average_prem, average_hypo, max_prem, max_hypo])
    merged = Dropout(DP)(merged)

    merged = Dense(SENT_REP_SIZE, activation = 'tanh')(merged)
    merged = Dropout(DP/2)(merged)
   
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
fnames = ['/data/s3094723/thesis/snli/snli_1.0/train.formated.1.sem', '/data/s3094723/thesis/snli/snli_1.0/train.formated.2.sem',
'/data/s3094723/thesis/snli/snli_1.0/dev.formated.1.sem', '/data/s3094723/thesis/snli/snli_1.0/dev.formated.2.sem',
'/data/s3094723/thesis/snli/snli_1.0/test.formated.1.sem','/data/s3094723/thesis/snli/snli_1.0/test.formated.2.sem']

tag_to_id = t2i(fnames)

training_tag1 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/train.formated.1.sem', MAX_LEN, tag_to_id)
training_tag2 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/train.formated.2.sem', MAX_LEN, tag_to_id)

validation_tag1 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/dev.formated.1.sem', MAX_LEN, tag_to_id)
validation_tag2 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/dev.formated.2.sem', MAX_LEN, tag_to_id)

test_tag1 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/test.formated.1.sem', MAX_LEN, tag_to_id)
test_tag2 = get_tag_data('/data/s3094723/thesis/snli/snli_1.0/test.formated.2.sem', MAX_LEN, tag_to_id)

#set variables
NO_TAGS = len(tag_to_id)
print('NO_TAGS: ', NO_TAGS)
VOCAB_LEN = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
TRAIN_EMBED = False
BATCH_SIZE = 128
PATIENCE = 4 # 8
MAX_EPOCHS = 42
DP = 0.2
OPTIMIZER = keras.optimizers.Adam(lr =  0.00005)
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
fmodel = Model(inputs=[premise_input, hypo_input], outputs=[preds, prem_out_tags, hypo_out_tags])
fmodel.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=[1., 0.01, 0.01])
fmodel.summary()

callbacks = [EarlyStopping(patience=PATIENCE)]
fmodel.fit([training[0],training[1]] , [training[2], training_tag1, training_tag2], 
         batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, 
         validation_data=([validation[0], validation[1]],  [validation[2], validation_tag1, validation_tag2]), callbacks = callbacks)


#loss, acc = fmodel.evaluate([test[0], test[1]], [test[2], test_tag1, test_tag2], batch_size=BATCH_SIZE)
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

all  = fmodel.evaluate([test[0], test[1]], [test[2], test_tag1, test_tag2], batch_size=BATCH_SIZE)

print(all)
