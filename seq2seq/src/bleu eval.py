# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:24:05 2018

@author: peter
"""
def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, StOS, reverse_target_char_index,
                    max_decoder_seq_length, EOS):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index[StOS]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_seq = []
    output_token_list = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        output_token_list.append(output_tokens)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_seq.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == EOS or
                len(decoded_seq) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    decoded_seq = np.array(decoded_seq)

    return decoded_seq, output_token_list


def decode_sequence_tree(input_seq, encoder_model, decoder_model, target_token_index, StOS, reverse_target_char_index,
                         EOS, f, threshold=None):
    if threshold is None:
        threshold = 1. / len(reverse_target_char_index)
    # Encode the input as state vectors.
    print('sequence tree threshold: {}'.format(threshold))

    encoder_states = encoder_model.predict(input_seq)  # encoder - decoder állapotok

    # Generate empty target sequence of length 1.
    input_token = np.zeros((1, 1))  # bemeneti token
    # Populate the first character of target sequence with the start character.
    input_token[0, 0] = target_token_index[StOS]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # stop_condition = False
    sequence_transitions = []  # predicted output, where [parent - child]
    full_seqs = list(input_token)  # teljes szekvencia
    transition_confidence = []  # tree transition weights
    sequence_depth = [0]  # szekvencia hosszúsága - fa mélysége
    states_tree = [encoder_states]
    transition_states = []
    # transition_states = states_tree
    parent_tokens = list(input_token)
    for i, input_token in enumerate(parent_tokens):
        output_tokens, h, c = decoder_model.predict(
            [np.array(input_token)] + states_tree[i])

        # Sample a token
        sampled_token_indices = np.where(output_tokens[0, -1, :] > threshold)[0]
        token_confidences = output_tokens[0, -1, :][list(sampled_token_indices.flatten())]

        [transition_confidence.append(confidence) for confidence in token_confidences]
        [transition_states.append(states_tree[i]) for _ in token_confidences]
        [sequence_depth.append(sequence_depth[i] + 1) for _ in range(len(sampled_token_indices))]
        [sequence_transitions.append(
            [input_token.flatten(), sampled_token.flatten(), sequence_depth[i]]) for
            sampled_token in sampled_token_indices]
        for sampled_token in sampled_token_indices:
            tmp_seq = list(full_seqs[i]) + [float(sampled_token)]
            full_seqs.append(np.array(tmp_seq))
        # [states_tree.append(states_value) for sampled_token in sampled_token_indices]
        # sampled_char = reverse_target_char_index[sampled_token_index]
        # decoded_seq.append(sampled_char)
        print(
            '{}. depth - {} is parent - res: {} - avgconf: {}'.format(sequence_depth[i], input_token,
                                                                      len(sampled_token_indices),
                                                                      np.average(token_confidences)))
        # Exit condition: either hit max length
        # or find stop character.
        for sampled_token in sampled_token_indices:
            sampled_char = reverse_target_char_index[sampled_token]
            if (sampled_char != EOS) and (sequence_depth[i] + 1 <= 5):
                parent_tokens.append(sampled_token.reshape((1, 1)))
                states_tree.append([h, c])
            else:
                pass

    return sequence_transitions, transition_confidence, transition_states, full_seqs


import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


INPUT_FILENAME = 'newstest-2015-100sents.en-ru.ref.ru'
OUTPUT_FILENAME = 'newstest-2015-100sents.en-ru.src.en'
REVERSE_SEQUENCES = True
TEST_SIZE = 0.0
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 1000
EMB_DIM = 40
LATENT_DIM = 256

with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
    input_sequences = f.readlines()
    
with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
    target_sequences = list(f.readlines())

input_sequences = [list(row) for row in input_sequences]
target_sequences = [list(row) for row in target_sequences]

# pad
input_max_len = max(len(row) for row in input_sequences)
target_max_len = max(len(row) for row in target_sequences)

for idx, row in enumerate(input_sequences):
    input_sequences[idx] += ' ' *  (input_max_len - len(row))
    
for idx, row in enumerate(target_sequences):
    target_sequences[idx] += ' ' * (target_max_len - len(row))
    
PAD = ' ' 
StOS = '$'
EOS = '#'

input_characters = np.append(np.array(input_sequences), [EOS, PAD, StOS])
input_characters = np.unique(input_characters)
target_characters = np.append(np.array(target_sequences), [EOS, PAD, StOS])
target_characters = np.unique(target_characters)

input_seqs = np.array(input_sequences)
target_seqs = np.array(target_sequences)
_t_seqs = []
for row in range(100):

    target_seq = target_sequences[row]
    target_len = input_max_len - np.where(target_seq[::-1] != ' ')[0][0] # the end of the sequence

    target_seq = np.insert(target_seq, 0, StOS)
    target_seq = np.insert(target_seq, target_len, EOS)
    _t_seqs.append(target_seq)

target_seqs = np.array(_t_seqs)

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(input_seq) for input_seq in input_seqs])
max_decoder_seq_length = max([len(target_seq) for target_seq in target_seqs])

print('Number of samples:', len(input_seqs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

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

# CREATING MODEL
# ENCODER
encoder_inputs = Input(shape=(None,), name='enc_input')
x = Embedding(num_encoder_tokens, EMB_DIM, name='enc_emb')(encoder_inputs)
x, state_h, state_c = LSTM(LATENT_DIM, return_state=True, name='enc')(x)
encoder_states = [state_h, state_c]

# DECODER
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,), name='dec_input')
emb = Embedding(num_decoder_tokens, EMB_DIM, name='dec_emb')
x = emb(decoder_inputs)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='dec')
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='dec_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   # Compile & run training
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([X_encoder_train, X_decoder_train], target_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=([X_encoder_test,X_decoder_test],target_test),
          verbose=1)

encoder_model = Model(encoder_inputs, encoder_states, name='enc_b')
encoder_model.summary()

decoder_state_input_h = Input(shape=(LATENT_DIM,), name='input_h')
decoder_state_input_c = Input(shape=(LATENT_DIM,), name='input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
x = emb(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    x, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states, name='dec_b')

decoder_model.summary()


reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

transitions, confidences, states, full_sequences = [], [], [], []

for input_seq in encoder_input_data:

    decoded_seq, _ = decode_sequence(input_seq, encoder_model, decoder_model,
                                                      target_token_index, StOS, reverse_target_char_index,
                                                      max_decoder_seq_length, EOS)

    seq_transitions, seq_confidences, state, seqs = decode_sequence_tree(input_seq[:10], encoder_model,
                                                                         decoder_model, target_token_index,
                                                                         StOS, reverse_target_char_index,
                                                                         max_decoder_seq_length, EOS)
    transitions.append(seq_transitions)
    confidences.append(seq_confidences)
    states.append(state)
    full_sequences.append(seqs)

val1 = []
val2 = []
val1s = []
val2s = []
for seq_index in range(len(encoder_input_data)):

    input_seq = encoder_input_data[seq_index]
    decoded_seq, _ = decode_sequence(input_seq, encoder_model, decoder_model,
                                                      target_token_index, StOS, reverse_target_char_index,
                                                      max_decoder_seq_length, EOS)

    decoded_seq = decoded_seq[:-1]
    orig_input_seq = input_seqs[seq_index]
    if PAD in target_seqs[seq_index]:
        length_sequence = list(target_seqs[seq_index]).index(PAD)
        true_seq = target_seqs[seq_index][1:length_sequence]
    else:
        length_sequence = list(target_seqs[seq_index]).index(EOS)
        true_seq = target_seqs[seq_index][1:]
    true_seq = true_seq[:-1]
    tmp = []
    for i in range(len(decoded_seq)):
        if decoded_seq[i] in true_seq:
            val1.append(seq_index)
            tmp.append(i)
    val2.append(len(tmp) / len(true_seq))
    val2s.append(len(tmp) / len(true_seq))
val1 = np.array(val1)
val1 = len(np.unique(val1)) / len(encoder_input_data)
val2 = np.mean(val2)
overall_results.append([val1, val2])


for fault in range(max(y_labels) + 1):
    fault_indices = np.where(y_labels == fault)[0]
    print(len(fault_indices))
    encoder_input_data_split = encoder_input_data[fault_indices]
    target_seqs_split = target_seqs[fault_indices]
    val1_tst = []
    val2_tst = []
    for seq_index in range(len(encoder_input_data_split)):
        input_seq = encoder_input_data_split[seq_index: seq_index + 1]
        decoded_seq, _ = decode_sequence(input_seq, encoder_model, decoder_model,
                                                          target_token_index, StOS, reverse_target_char_index,
                                                          max_decoder_seq_length, EOS)

        decoded_seq = decoded_seq[:-1]
        if PAD in target_seqs_split[seq_index]:
            length_sequence = list(target_seqs_split[seq_index]).index(PAD)
            true_seq = target_seqs_split[seq_index][1:length_sequence]
        else:
            length_sequence = list(target_seqs_split[seq_index]).index(EOS)
            true_seq = target_seqs_split[seq_index][1:]
        true_seq = true_seq[:-1]
        tmp = []
        for i in range(len(decoded_seq)):
            if decoded_seq[i] in true_seq:
                val1_tst.append(seq_index)
                tmp.append(i)
        val2_tst.append(len(tmp) / len(true_seq))
    val1_tst = np.array(val1_tst)
    if len(encoder_input_data_split) != 0:
        val1_tst = len(np.unique(val1_tst)) / len(encoder_input_data_split)
    else:
        val1_tst = 0
    val2_tst = np.mean(val2_tst)
    results_val1.append(val1_tst)
    results_val2.append(val2_tst)

results_val1 = np.array(results_val1)
results_val2 = np.array(results_val2)
overall_results = np.array(overall_results)
