import numpy as np
import pandas as pd
import math
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding #, TimeDistributed
from sklearn.model_selection import train_test_split #, StratifiedShuffleSplit
#import os


#%% Params for dataset

# datasets
FILENAME = '..\\dataset\\Sequences_alarm_warning_171208.xlsx'
KOSARAK = 'kosarak25k.txt'
FIFA = 'FIFA.txt'
BIBLE = 'BIBLE.txt'
BMS2 = 'BMS2.txt'

# confidences, if given
SHEET = ''
CONFIDENCES_SHEET = None

# temporal setting
WITHOUT_TEMPORAL = True

# fault label for crossval
FAULT_LABELS_FILENAME = ''

# sequence split settings
SEQUENCE_MIN_LENGTH = 1 # sequences with length under given min_length will be dropped
INPUT_LENGTH = None # if given, all sequences will have fixed input length
SPLIT_WITH_CONFIDENCES = False

REVERSE_SEQUENCES = True

def split_data_with_confidences(threshold=0.2):

    if CONFIDENCES_SHEET is None:
        confs = get_sequence_confidences(np.append(input_seqs, target_seqs[:, 1:], axis=1))
    else:
        confs = pd.read_excel(FILENAME, header=None, dtype=float, sheet_name=CONFIDENCES_SHEET)
        confs = confs.as_matrix()

    trans_conf = confs[:, -1][:, np.newaxis] / confs
    input_length = input_seqs.shape[1]
    target_length = target_seqs.shape[1]
    new_input_seqs, new_target_seqs, new_labels, confidence_splits, confidences = [], [], [], [], []

    if input_length != target_length:
        longest_seq = input_length
    else:
        longest_seq = 0

    for idx in range(len(input_seqs)):

        where_to_split = np.where(trans_conf[idx][:input_length] > threshold)[0]
        for split in where_to_split.tolist()[1:]:

            tmp_input_seq = input_seqs[idx][:split].tolist()
            longest_seq = max(input_length - split, longest_seq)
            tmp_input_seq += [PAD] * (input_length - split)
            tmp_target_seq = [StOS]
            tmp_target_seq += input_seqs[idx][split:].tolist()
            tmp_target_seq[tmp_target_seq.index(PAD):] = target_seqs[idx][1:target_seqs[idx].tolist().index(PAD)].tolist()
            tmp_target_seq += [PAD] * (target_length - len(tmp_target_seq))

            new_input_seqs.append(tmp_input_seq)
            new_target_seqs.append(tmp_target_seq)
            new_labels.append(classification_labels[idx])
            confidence_splits.append(trans_conf[idx, split])
            confidences.append(trans_conf[idx, :split])
            

        new_input_seqs.append(input_seqs[idx])
        new_target_seqs.append(target_seqs[idx])
        new_labels.append(classification_labels[idx])
        confidence_splits.append(trans_conf[idx, trans_conf.shape[1] - 1])
        confidences.append(trans_conf[idx])

    return np.array(new_input_seqs), np.array(new_target_seqs), new_labels, np.array(confidence_splits), confidences


def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, StOS, reverse_target_char_index,
                    max_decoder_seq_length, EOS):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    input_token = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    input_token[0, 0] = target_token_index[StOS]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sequences = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [input_token] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_characters = reverse_target_char_index[sampled_token_index]
        decoded_sequences.append(sampled_characters)

        # Exit condition: either hit max length
        # or find stop character.
        if (int(sampled_characters) == int(EOS) or
                len(decoded_sequences) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        input_token = np.zeros((1, 1))
        input_token[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    decoded_sequences = np.array(decoded_sequences)

    return decoded_sequences


def decode_sequence_tree(input_seq, encoder_model, decoder_model, target_token_index, StOS, reverse_target_char_index,
                         EOS, f, threshold=None):
    if threshold is None:
        threshold = False
    # Encode the input as state vectors.
    print('sequence tree threshold: {}'.format(threshold))
    print('input data: {}'.format(input_seq.astype(int)))
    encoder_states = encoder_model.predict(input_seq)  # encoder - decoder állapotok

    # Generate empty target sequence of length 1.
    input_token = np.zeros((1, 1))  # bemeneti token
    # Populate the first character of target sequence with the start character.
    input_token[0, 0] = target_token_index[StOS]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # stop_condition = False
    sequence_transitions = []  # predicted output, where [parent - child - max, depth]
    full_seqs = list(input_token)  # teljes szekvencia
    transition_probabilities = []  # tree transition probabilities
    sequence_depth = [0]  # szekvencia hosszúsága - fa mélysége
    states_tree = [encoder_states]
    transition_states = []
    # transition_states = states_tree
    parent_tokens = list(input_token)
    for i, input_token in enumerate(parent_tokens):
        
        if reverse_target_char_index[float(input_token)] == PAD:
            pass
        else:
                
            not_sampled = True
            while not_sampled:
            
                output_tokens, h, c = decoder_model.predict(
                    [np.array(input_token)] + states_tree[i])
    
            # Sample a token
                if threshold:
                    
                    sampled_token_indices = np.where(output_tokens[0, -1, :] > threshold)[0]
                    max_probable_token_index = np.argmax(output_tokens[0, -1, :])
                    if len(sampled_token_indices) > 0:
                        not_sampled = False
                    else:
                        threshold *= 0.9
                
                else:
                    
                    sample_threshold = np.max(output_tokens[0, -1, :])*1
                    max_probable_token_index = np.argmax(output_tokens[0, -1, :])
                    sampled_token_indices = np.where(output_tokens[0, -1, :] >= sample_threshold)[0]
                    not_sampled = False
                    
            
            token_probabilities = output_tokens[0, -1, :][list(sampled_token_indices.flatten())]
    
            [transition_probabilities.append(confidence) for confidence in token_probabilities]
            [transition_states.append(states_tree[i]) for _ in token_probabilities]
            [sequence_depth.append(sequence_depth[i] + 1) for _ in range(len(sampled_token_indices))]
            [sequence_transitions.append(
                [input_token.flatten(), sampled_token.flatten(), sequence_depth[i]]) for
                sampled_token in sampled_token_indices]
            for sampled_token in sampled_token_indices:
                tmp_seq = list(full_seqs[i]) + [float(sampled_token)]
                full_seqs.append(np.array(tmp_seq, dtype='int32'))
            # [states_tree.append(states_value) for sampled_token in sampled_token_indices]
            # sampled_char = reverse_target_char_index[sampled_token_index]
            # decoded_seq.append(sampled_char)
            print(
                '{}. depth - {} -> max_probability: {}({}) - intersects to : {}'.format(sequence_depth[i], int(input_token),
                                                                          int(max_probable_token_index),
                                                                          np.max(token_probabilities),len(sampled_token_indices)))
            # Exit condition: either hit max length
            # or find stop character.
            for sampled_token in sampled_token_indices:
                parent_tokens.append(sampled_token.reshape((1, 1)))
                states_tree.append([h, c])

    return sequence_transitions, transition_probabilities, transition_states, full_seqs


def decode_sequence_tree_2(input_seq, encoder_model, decoder_model, target_token_index, StOS, reverse_target_char_index,
                         EOS, f, threshold=None):
    if threshold is None:
        threshold = False
    # Encode the input as state vectors.
    print('sequence tree threshold: {}'.format(threshold))
    print('input data: {}'.format(input_seq.astype(int)))
    encoder_states = encoder_model.predict(input_seq)  # encoder - decoder állapotok

    # Generate empty target sequence of length 1.
    input_token = np.zeros((1, 1))  # bemeneti token
    # Populate the first character of target sequence with the start character.
    input_token[0, 0] = target_token_index[StOS]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # stop_condition = False
    sequence_transitions = []  # predicted output, where [parent - child - max, depth]
    full_seqs = list(input_token)  # teljes szekvencia
    transition_probabilities = []  # tree transition probabilities
    sequence_depth = [0]  # szekvencia hosszúsága - fa mélysége
    states_tree = [encoder_states]
    transition_states = []
    # transition_states = states_tree
    parent_tokens = list(input_token)
    for i, input_token in enumerate(parent_tokens):
        
        if sequence_depth[-1] > 5:
            break
        
        if reverse_target_char_index[float(input_token)] == PAD:
            pass
        else:
                
            not_sampled = True
            while not_sampled:
            
                output_tokens = decoder_model.predict(
                    [np.array(input_token)] + states_tree[i])
    
            # Sample a token
                if threshold:
                    
                    sampled_token_indices = np.where(output_tokens[0, -1, :] > threshold)[0]
                    max_probable_token_index = np.argmax(output_tokens[0, -1, :])
                    if len(sampled_token_indices) > 0:
                        not_sampled = False
                    else:
                        threshold *= 0.9
                
                else:
                    
                    sample_threshold = np.max(output_tokens[0, -1, :])*1
                    max_probable_token_index = np.argmax(output_tokens[0, -1, :])
                    sampled_token_indices = np.where(output_tokens[0, -1, :] >= sample_threshold)[0]
                    not_sampled = False
                    
            
            token_probabilities = output_tokens[0, -1, :][list(sampled_token_indices.flatten())]
    
            [transition_probabilities.append(confidence) for confidence in token_probabilities]
            [transition_states.append(states_tree[i]) for _ in token_probabilities]
            [sequence_depth.append(sequence_depth[i] + 1) for _ in range(len(sampled_token_indices))]
            [sequence_transitions.append(
                [input_token.flatten(), sampled_token.flatten(), sequence_depth[i]]) for
                sampled_token in sampled_token_indices]
            for sampled_token in sampled_token_indices:
                tmp_seq = list(full_seqs[i]) + [float(sampled_token)]
                full_seqs.append(np.array(tmp_seq, dtype='int32'))
            # [states_tree.append(states_value) for sampled_token in sampled_token_indices]
            # sampled_char = reverse_target_char_index[sampled_token_index]
            # decoded_seq.append(sampled_char)
            print(
                '{}. depth - {} -> max_probability: {}({}) - intersects to : {}, input_seq: {}'.format(sequence_depth[i], int(input_token),
                                                                          int(max_probable_token_index),
                                                                          np.max(token_probabilities),len(sampled_token_indices), input_seq))
            # Exit condition: either hit max length
            # or find stop character.
            for sampled_token in sampled_token_indices:
                parent_tokens.append(sampled_token.reshape((1, 1)))
                input_seq = input_seq[1:]
                input_seq = np.concatenate((np.zeros(1), input_seq), axis=0)
                input_seq[max(np.where(input_seq == target_token_index[PAD])[0])] = sampled_token 
                
                encoder_states = encoder_model.predict(input_seq)
                states_tree.append(encoder_states)

    return sequence_transitions, transition_probabilities, transition_states, full_seqs



def get_sequence_confidences(sequence_data):

    from evaluate import get_conf
    from tqdm import tqdm

    confs = []
    for row in tqdm(sequence_data[:, 1:]):

        seq_confs = []
        for e in range(len(row)):
            seq_confs.append(get_conf(row[:e + 1], sequence_data[:, 1:]))
        confs.append(seq_confs)

    confs = np.array(confs)

    return confs
#%% Params for training and model

TEST_SIZE = 0.0
RANDOM_STATE = 42
BATCH_SIZE = 1
EPOCHS = 1000

EMB_DIM = 4
LATENT_DIM = 512

# open gyógypélda
with open('gyogypelda.txt', 'r') as f:
    seq = f.readlines()
seq = [line.replace('\n', '').split(' ')[:] for line in seq]
seq = seq[:-1]
seq = [[float(item) for item in line] for line in seq]
DS_EOS = min(seq[0])
DS_StOS = DS_EOS - 1
DS_PAD = DS_EOS - 2
seq_max_len = max(len(row) for row in seq)
for idx, row in enumerate(seq):
    seq[idx] += [DS_PAD] *  (seq_max_len - len(row) + 1)
seq = np.array(seq)
seq_input_c= np.append(seq, [DS_EOS, DS_PAD, DS_StOS])
seq_input_c=seq_target_c= np.unique(seq_input_c)
sequences = seq
input_characters, target_characters = seq_input_c, seq_target_c
max_length = seq_max_len
PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS
input_seqs = []
target_seqs = []
classification_labels = []

#%%
for row in range(sequences.shape[0]):

    sequence_length = list(sequences[row]).index(PAD)  # the end of the sequence
    if sequence_length > SEQUENCE_MIN_LENGTH:

        if not INPUT_LENGTH:
            input_seq = sequences[row, :math.floor(sequence_length / 2)]
            input_seq = np.array(input_seq.tolist() + (max_length - input_seq.shape[0]) * [PAD])
            target_seq = sequences[row, math.floor(sequence_length / 2):]
            target_seq = np.array(target_seq.tolist() + (max_length - target_seq.shape[0]) * [PAD])
        else:
            input_seq = sequences[row, :INPUT_LENGTH]
            target_seq = sequences[row, INPUT_LENGTH:]

        target_seq = np.insert(target_seq, 0, StOS)
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

    classification_labels.append(row)

input_seqs = np.array(input_seqs)
target_seqs = np.array(target_seqs)

#%% split where confidence is high enough 
if SPLIT_WITH_CONFIDENCES or CONFIDENCES_SHEET:
    input_seqs, target_seqs, classification_labels, conf_splits, confs = split_data_with_confidences()
else:
    conf_splits, confs = None, None

#%%
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(input_seq) for input_seq in input_seqs])
max_decoder_seq_length = max([len(target_seq) for target_seq in target_seqs])

print('Number of samples:', len(input_seqs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

#%% Define LSTM based RNN model


encoder_inputs = Input(shape=(None,), name='encoder_input')
encoder_embedding = Embedding(num_encoder_tokens, EMB_DIM, name='encoder_embedding')(encoder_inputs)
encoder = LSTM(LATENT_DIM, stateful=False, return_sequences=True, return_state=True, name='encoder')
encoder_outputs, state_h, state_c = encoder(encoder_embedding)
encoder_states = [state_h, state_c]

#-----------------------------------------
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, EMB_DIM, name='decoder_embedding')(decoder_inputs)

decoder_1 = LSTM(LATENT_DIM, stateful=False, return_sequences=True, return_state=False, name='decoder_1', dropout=0.2, recurrent_dropout=0.2)
decoder_2 = LSTM(LATENT_DIM, stateful=False, return_sequences=True, return_state=True, name='decoder_2', dropout=0.2, recurrent_dropout=0.2)

decoder_outputs, _, _ = decoder_2(decoder_1(decoder_embedding, initial_state=encoder_states))
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

#-------------------------------------------
train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_model.summary()
#%% 

# We do EPOCHS time training over the dataset, where we load input_split size batches for training
# only for one epoch, then load the next batch for one epoch. One epoch is indicated by epoch_id, where
# we do numerous memory load
  

input_splits = list(range(5000,int(len(input_seqs)), 5000))
lower_input_splits = list(range(0,int(len(input_seqs))-5000,5000))


input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = input_token_index

#%% Fill encoder-, decoder input data, target data

encoder_input_data = np.zeros((len(input_seqs), max_encoder_seq_length), dtype='float32')
decoder_input_data = np.zeros((len(input_seqs), max_decoder_seq_length), dtype='float32')
decoder_target_data = np.zeros((len(input_seqs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        
for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
    for t, char in enumerate(input_seq):
        if REVERSE_SEQUENCES:
            encoder_input_data[i, -(t + 1)] = input_token_index[char]
        else:
            encoder_input_data[i, t] = input_token_index[char]
    for t, char in enumerate(target_seq):
        # decoder_target_data is ahead of decoder_input_data by one timestep

        decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
        
        
X_encoder_train, X_encoder_test, X_decoder_train, X_decoder_test, target_train, target_test = train_test_split(
encoder_input_data, decoder_input_data, decoder_target_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
            
#%% Train to encode and decode
train_model.fit([X_encoder_train, X_decoder_train], target_train,
  batch_size=BATCH_SIZE,
  epochs=1024,
  verbose=2)

#%% build encoder model

encoder_model = Model(encoder_inputs, encoder_states, name='enc_b')
encoder_model.summary()

#%% build decoder model

decoder_state_input_h = Input(shape=(LATENT_DIM,), name='input_h')
decoder_state_input_c = Input(shape=(LATENT_DIM,), name='input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs,_,_ = decoder_2(decoder_1(decoder_embedding,
                                                    initial_state = decoder_states_inputs))
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs], name='dec_b')

decoder_model.summary()


 #%%

results_val1, results_val2, overall_results = [], [], []
#decoder_model.load_weights('kosar_decoder_w.h5')
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# transitions: in which we store the tree structure, which has [[parent], [child], [depth in tree]]

transitions, probabilities, states, full_sequences = [], [], [], []

for input_seq in encoder_input_data[:10]:
    """
    decoded_seq = decode_sequence(input_seq, encoder_model, decoder_model,
                                                      target_token_index, StOS, reverse_target_char_index,
                                                      max_decoder_seq_length, EOS)
    """
    seq_transitions, seq_probabilities, state, seqs = decode_sequence_tree_2(input_seq, encoder_model,
                                                                           decoder_model, target_token_index,
                                                                           StOS, reverse_target_char_index,
                                                                           max_decoder_seq_length, EOS)
    transitions.append(seq_transitions)
    probabilities.append(seq_probabilities)
    states.append(state)
    full_sequences.append(seqs)