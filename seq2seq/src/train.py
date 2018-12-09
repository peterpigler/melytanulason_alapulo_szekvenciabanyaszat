from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np
import os

cwd = os.getcwd()
files = os.listdir(cwd)


def train_sequences_crossval(encoder_input_data, decoder_input_data, decoder_target_data,
                             input_token_index, target_token_index,
                             input_seqs, target_seqs, faultclass, EOS, PAD, StOS, max_decoder_seq_length,
                             num_encoder_tokens, num_decoder_tokens,
                             emb_dim=40, latent_dim=256, batch_size=256, epochs=4000,
                             n_splits=7, test_size=0.2, random_state=42,
                             modelname=None, threshold=None):

    SSS = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    results_val1 = []
    results_val2 = []
    models = []
    overall_results = []
    overall_results_crossval = []
    for train_index, test_index in SSS.split(encoder_input_data, faultclass):

        encoder_inputs = Input(shape=(None,), name='enc_input')
        x = Embedding(num_encoder_tokens, emb_dim, name='enc_emb')(encoder_inputs)
        x, state_h, state_c = LSTM(latent_dim,
                                   return_state=True, name='enc')(x)
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name='dec_input')
        emb = Embedding(num_decoder_tokens, emb_dim, name='dec_emb')
        x = emb(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='dec')
        decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='dec_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.summary()
        # plot_model(model, to_file='model.png', show_shapes =True)
        # Compile & run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # Note that `decoder_target_data` needs to be one-hot encoded,
        # rather than sequences of integers like `decoder_input_data`!

        model.fit([encoder_input_data[train_index], decoder_input_data[train_index]], decoder_target_data[train_index],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=([encoder_input_data[test_index],
                                    decoder_input_data[test_index]
                                    ],
                                   decoder_target_data[test_index]),
                  verbose=1)

        # model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_model_12_15_1422.h5')
        # Save model
        if modelname:
            model.save(modelname + '.h5')

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states, name='enc_b')
        # The encoder model is defined as taking the input layer from the encoder
        # in the trained model (encoder_inputs) and outputting the hidden and
        # cell state tensors (encoder_states).
        # plot_model(encoder_model, to_file='model_enc.png', show_shapes =True)
        encoder_model.summary()
        # encoder_model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_encoder_12_15_1422.h5')
        decoder_state_input_h = Input(shape=(latent_dim,), name='input_h')
        decoder_state_input_c = Input(shape=(latent_dim,), name='input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        x = emb(decoder_inputs)
        decoder_outputs, state_h, state_c = decoder_lstm(
            x, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states, name='dec_b')
        # plot_model(decoder_model, to_file='model_dec.png', show_shapes =True)
        decoder_model.summary()
        # decoder_model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_decoder_12_15_1422.h5')
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        analysed_seqs = np.arange(len(input_seqs))

        seq_trans, seq_cons, states, full_seqs = [], [], [], []

        for num in range(len(analysed_seqs)):
            # Take one sequence (part of the training test)
            # for trying out decoding.
            seq_index = analysed_seqs[num]
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_seq, _ = decode_sequence(input_seq, encoder_model, decoder_model,
                                                              target_token_index, StOS, reverse_target_char_index,
                                                              max_decoder_seq_length, EOS)

            seq_transitions, seq_confidences, state, seqs = decode_sequence_tree(input_seq[:10], encoder_model,
                                                                                 decoder_model, target_token_index,
                                                                                 StOS, reverse_target_char_index,
                                                                                 max_decoder_seq_length, EOS, 0.20)
            seq_trans.append(seq_transitions)
            seq_cons.append(seq_confidences)
            states.append(state)
            full_seqs.append(seqs)
            print('-')
            print('Input sequence:', input_seqs[seq_index])
            print('Decoded sequence:', decoded_seq)
            if PAD in target_seqs[seq_index]:
                length_sequence = list(target_seqs[seq_index]).index(PAD)
                print('True sequence:', target_seqs[seq_index][1:length_sequence])
            else:
                length_sequence = list(target_seqs[seq_index]).index(EOS)
                print('True sequence:', target_seqs[seq_index][1:])

        val1 = []
        val2 = []
        val1s = []
        val2s = []
        for seq_index in range(len(encoder_input_data)):
            input_seq = encoder_input_data[seq_index: seq_index + 1]
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
        val1s.append(np.unique(val1))
        val1 = len(np.unique(val1)) / len(encoder_input_data)
        val2 = np.mean(val2)
        overall_results.append([val1, val2])
        overall_results_crossval = np.append(overall_results_crossval, [val1s, val2s])

        result_1val = []
        result_2val = []
        for fault in range(max(faultclass) + 1):
            fault_indices = np.where(faultclass == fault)[0]
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
            result_1val.append(val1_tst)
            result_2val.append(val2_tst)

        results_val1.append(result_1val)
        results_val2.append(result_2val)

    results_val1 = np.array(results_val1)
    results_val2 = np.array(results_val2)
    overall_results = np.array(overall_results)

    return results_val1, results_val2, overall_results, seq_trans, seq_cons, states, full_seqs


def train_sequences(encoder_input_data, decoder_input_data, decoder_target_data,
                             input_token_index, target_token_index,
                             input_seqs, target_seqs, faultclass, EOS, PAD, StOS, max_decoder_seq_length,
                             num_encoder_tokens, num_decoder_tokens,
                             emb_dim=40, latent_dim=256, batch_size=256, epochs=4000, test_size=0.2, random_state=42,
                             modelname=None, threshold=None):

    X_encoder_train, X_encoder_test, X_decoder_train, X_decoder_test, target_train, target_test = train_test_split(
        encoder_input_data, decoder_input_data, decoder_target_data, test_size=test_size, random_state=random_state)
    results_val1, results_val2, overall_results = [], [], []

    # CREATING MODEL
    # ENCODER
    encoder_inputs = Input(shape=(None,), name='enc_input')
    x = Embedding(num_encoder_tokens, emb_dim, name='enc_emb')(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim,
                               return_state=True, name='enc')(x)
    encoder_states = [state_h, state_c]

    # DECODER
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,), name='dec_input')
    emb = Embedding(num_decoder_tokens, emb_dim, name='dec_emb')
    x = emb(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='dec')
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
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([X_encoder_test,X_decoder_test],target_test),
              verbose=1)

    encoder_model = Model(encoder_inputs, encoder_states, name='enc_b')
    encoder_model.summary()

    decoder_state_input_h = Input(shape=(latent_dim,), name='input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_c')
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
                                                                             max_decoder_seq_length, EOS, 0.20)
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


    for fault in range(max(faultclass) + 1):
        fault_indices = np.where(faultclass == fault)[0]
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

    return results_val1, results_val2, overall_results, transitions, confidences, states, full_sequences


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
                         EOS, threshold=None):
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


import dataset
import os

os.getcwd()

input_seqs, target_seqs, sequences, input_characters, [PAD, EOS, StOS], y_labels, confs_splits = dataset.load_sequences(
    '..\\dataset\\MOL_adatbazis\\Uzemseq.xlsx', sheet='seqs', length_min_seq=5, confidences='confs')
PADfaultclass, y_label_array, [YEncoder, YEncoder_OH] = dataset.decode_faultclasses('..//dataset//Fault_0822.xlsx', y_labels)
# input_seqs, target_seqs, y_label = dataset.split_sequences_with_labels(input_seqs, target_seqs, sequences, 5, 3, PAD, StOS, EOS, y_label)
target_characters = input_characters

encoder_input_data, decoder_input_data, decoder_target_data, [num_encoder_tokens, num_decoder_tokens], [
    max_encoder_seq_length, max_decoder_seq_length], [input_token_index, target_token_index] = dataset.tokenize(
    input_seqs, target_seqs, input_characters, target_characters)

"""
resval1, resval2, ovres, [output_token_list0, output_token_list1,
                          output_token_list2], tree_seq, tree_conf, states, seqs = train_sequences_crossval(
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, input_seqs,
    target_seqs, classification_labels, EOS, PAD, StOS, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, 40, 256, 256,
    100, n_splits=1, test_size=0.2, random_state=42, modelname=None)
"""

resval1, resval2, ovres, tree_seq, tree_conf, states, seqs = train_sequences(encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, input_seqs,
    target_seqs, y_labels, EOS, PAD, StOS, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, 40, 256, 256,
    1000, test_size=0.2, random_state=42, modelname=None)


import pandas as pd
import evaluate

# Create Tree dictionaries
tree_dicts = []
for idx, tree in enumerate(tree_seq):
    try:
        tree_dicts.append(evaluate.build_dictionary(tree, tree_conf[idx]))
    except:
        print('Tree doesn\'t contain enough node: {}.'.format(idx))

# Build Trees
tree_objects = []
chain_conf = []
for one_tree, one_tree_conf in zip(seqs, tree_conf):
    tmp_tree, tmp_chain_conf = evaluate.build_tree(one_tree, one_tree_conf)
    chain_conf.append(tmp_chain_conf)
    tree_objects.append(tmp_tree)

# Find occurances 
sequence_occurances = []
for one_tree in seqs:
    sequence_occurances.append(evaluate.find_frequency(one_tree, decoder_input_data))

transitions = np.array(tree_seq[0],dtype=int)[:,:2]
for idx, one_tree in enumerate(tree_seq[1:]):
    try:
        transitions = np.vstack((transitions,np.array(one_tree,dtype=int)[:,:2]))
    except:
        print('Tree doesn\'t contain enough node: {}.'.format(idx))

"""
states_tmp = np.array(states[0])
states_tmp = states_tmp.reshape((len(states_tmp),2,256))
states_tmp = states_tmp[:,0,:] # h
for state in states[1:]:
    state = np.array(state)
    state = state.reshape((len(state),2,256))
    states_tmp = np.vstack((states_tmp,state[:,0,:]))
states = states_tmp

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
h_pca = pca.fit_transform(states)

plt.scatter(h_pca[:,0], h_pca[:,1])
for i, trans in enumerate(transitions):
    plt.annotate(trans, (h_pca[i,0],h_pca[i,1]))
plt.show()
"""
"""
depths = np.array(tree_seq[5], dtype=int)[:,2]
transitions = np.array(tree_seq[5],dtype=int)[:,:2]
counts = np.max(np.bincount(np.array(depths)))

dct = {}
for depth in list(set(depths)):
    dct[depth] = []
for depth in list(set(depths)):
    dct['{} confidents'.format(depth)] = []
for idx, depth in enumerate(depths):
    dct[depth].append(transitions[idx])
    dct['{} confidents'.format(depth)].append(tree_conf[5][idx])
for depth in list(set(depths)):
    dct[depth] += [''] * (counts-len(dct[depth]))
    dct['{} confidents'.format(depth)] += [''] * (counts-len(dct['{} confidents'.format(depth)]))

df = pd.DataFrame.from_dict(dct)
df.to_csv('one_tree_2.csv')

one_tree = seqs[8]
tree_states = states[8]
tree_states = np.array(tree_states[8])
tree_states = tree_states.reshape((len(tree_states),2,256))
h=tree_states[:,0,:]
c=tree_states[:,1,:]

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
h_pca = pca.fit_transform(h)
c_pca = pca.fit_transform(c)

plt.scatter(h_pca[:,0], h_pca[:,1])
plt.scatter(c_pca[:,0], c_pca[:,1])
plt.plot([h_pca[:,0], c_pca[:,0]],[h_pca[:,1], c_pca[:,1]],'k-', lw=0.1)
pos = (h_pca + c_pca) / 2.
for i, trans in enumerate(transitions):
    plt.annotate(trans, (pos[i,0],pos[i,1]))
plt.show()


probabilities = tree_conf[8]
probabilities = np.insert(probabilities, 0, 1.0)

from ete3 import Tree, TreeStyle

tree = Tree()
nodes = {'': [1.0, tree]}
for idx, seq in enumerate(one_tree):
    parent = ''
    for i in range(len(seq)): 
        if not str(seq[:i+1]) in tree:
            nodes[str(seq[:i+1])] = []
            nodes[str(seq[:i+1])].append(probabilities[idx] * nodes[parent][0]) 
            nodes[str(seq[:i+1])].append(nodes[parent][1].add_child(name=str(seq), dist=nodes[str(seq[:i+1])][0]))
            #nodes[str(seq[:i+1])] = nodes[child].add_feature('confidence', tree_conf[19][idx])
        else:
            parent = str(seq[:i+1])

ts = TreeStyle()
ts.show_leaf_name = True
ts.show_branch_length = True
ts.show_branch_support = True
print(tree)

def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

occurances = []
for generated_sequence in one_tree:

    occurance = 0
    for decoder_data in decoder_input_data:
        if np.sum(np.all(rolling_window(decoder_data,len(generated_sequence)) == generated_sequence, axis=1)):
            occurance += 1
    occurances.append(occurance / len(decoder_input_data))
"""
print('finished..')
