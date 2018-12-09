#%% Import
import numpy as np
import pandas as pd
import math
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding #, TimeDistributed
from sklearn.model_selection import train_test_split #, StratifiedShuffleSplit
#import os

#%% Fájlok
FILENAME = '..\\dataset\\Sequences_alarm_warning_171208.xlsx'
KOSARAK = 'kosarak25k.txt'
FIFA = 'FIFA.txt'
BIBLE = 'BIBLE.txt'
BMS2 = 'BMS2.txt'

#%% Konfidenciaszámításhoz létező adatok
# confidences, if given
SHEET = '' # ha sheet='', akkor nem keresi excel munkalapon, különben sheet-en
CONFIDENCES_SHEET = None 

# temporal setting
WITHOUT_TEMPORAL = True # Ha az adatok temporállal nélkül vannak, true, egyébként false-szal kiszedi  

# fault label for crossval
FAULT_LABELS_FILENAME = '' # Ha a szekvenciákhoz rendelhető valamilyen osztály, akkor ebben a fájlban keresse a címkéket

# sequence split settings
SEQUENCE_MIN_LENGTH = 1 # sequences with length under given min_length will be dropped
INPUT_LENGTH = None # if given, all sequences will have fixed input length
SPLIT_WITH_CONFIDENCES = False # számítson hozzá konfidenciát, és a magas konfidenciájú értékeknél vágjon egyet

REVERSE_SEQUENCES = True # a bemenő szekvencia iránya fordított legyen-e

#%% Függvények
def split_data_with_confidences(threshold=0.2):
    
    if CONFIDENCES_SHEET is None:
        confs = get_sequence_confidences(np.append(input_seqs, target_seqs[:, 1:], axis=1)) # Ha számítsunk konfidenciát...
    else:
        confs = pd.read_excel(FILENAME, header=None, dtype=float, sheet_name=CONFIDENCES_SHEET)
        confs = confs.as_matrix()
    
    trans_conf = confs[:, -1][:, np.newaxis] / confs # átmenet konfidenciája
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


def decode_sequence(input_seq, threshold=0.8):
        """
        generation steps:
            1. decode states from encoder_model
            2. create target_token
            3. keep track of states (states_list), tokens (token_list),
                sequence depth (depth_in_predict), probabilities (token_prob_list), 
                max probability sequence (decoded_seq)
            4. loop:
                4.1. check if input is valid (EOS or PAD)
                4.2. predict next token in sequence using states (states_list[i]), 
                    input token (token_list[i])
                4.3. get hightest prob. token (argmax -> sampled_token_index), 
                    and probability (max - max_token_probability)
                4.4. reverse the token (reverse_target_char_index -> sampled_token)
                4.5. if sampled_token_index in decoded_sequence (max prob.), 
                    append to decode_sequence
                4.6. push (if output is valid) states (states_list), 
                    tokens (token_list), 
                    sequence depth (depth_in_predict), 
                    probabilities (token_prob_list)
        """
        
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
        
        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_token_index[StOS]
        	
        
        # keep track of tokens (token_list), states (states_list) 
        # and sequence depth/length (prediction_depth)
        token_list = [target_seq]
        states_list = [states_value]
        prediction_depth = [0]
        prediction_probabilities = [1.0]
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        	
        decoded_seq = []
        for states, token in zip(states_list, token_list):
            
            output_tokens, h, c = decoder_model.predict(
            [token] + states)
        
            # Sample a token
            probability = np.max(output_tokens[0, -1, :])
            max_sampled_token_index = np.argmax(output_tokens[0, 0])
            sampled_token_indices = np.where(output_tokens[0, 0] > probability * threshold)[0]
            
            max_sampled_char = reverse_target_char_index[max_sampled_token_index]
            sampled_chars = []
            [sampled_chars.append(reverse_target_char_index[token]) for token in sampled_token_indices]
            decoded_seq.append(max_sampled_char)
            		
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            
            for token in sampled_token_indices:
                
                if (reverse_target_char_index[token] != EOS and prediction_depth[-1] < max_decoder_seq_length):
                    target_seq[0, 0] = token
                    token_list.append(target_seq)
                    # Update states
                    states_value = [h, c]
                    states_list.append(states_value)
                    prediction_depth.append(prediction_depth[-1] + 1)
                    prediction_probabilities.append(output_tokens[0, -1, :][token])
            		
            # Exit condition: either hit max length
            # or find stop character.
            			
        decoded_seq = np.array(decoded_seq)
        
        return decoded_seq, token_list, prediction_depth, prediction_probabilities


def decode_sequence_tree(input_seq, encoder_model, decoder_model, target_token_index, StOS, reverse_target_char_index,
                         EOS, threshold=None):
    if threshold is None:
        threshold = False
    # Encode the input as state vectors.
    print('sequence tree threshold: {}'.format(threshold))
    encoder_states = encoder_model.predict(input_seq)  # encoder - decoder állapotok
    input_seq = [int(reverse_target_char_index[token]) for token in list(input_seq)]
    print('input data: {}'.format(input_seq))
    
    
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
        
        if reverse_target_char_index[float(input_token)] == EOS:
            #print('bent-EOS!!')
            pass
        elif reverse_target_char_index[float(input_token)] == PAD:
            #print('{}bent-pad'.format(input_token))
            pass
        else:
            #print('{}valami'.format(reverse_target_char_index[float(input_token)]))
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
                    
                    sample_threshold = np.max(output_tokens[0, -1, :])*1 # a legnagyobb értéket veszi ki, ha nem 1 -> threshold
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
                '{}. depth:\t{} >> max_probability:\t{}\t({}) - intersects to : {}'.format(sequence_depth[i], int(reverse_target_char_index[float(input_token)]),
                                                                          int(reverse_target_char_index[float(max_probable_token_index)]),
                                                                          np.max(token_probabilities),len(sampled_token_indices)))
            # Exit condition: either hit max length
            # or find stop character.
            for sampled_token in sampled_token_indices:
                parent_tokens.append(sampled_token.reshape((1, 1)))
                states_tree.append([h, c])
    
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

#%% Paraméterek
TEST_SIZE = 0.0
RANDOM_STATE = 42
BATCH_SIZE = 512
EPOCHS = 256

EMB_DIM = 40
LATENT_DIM = 256


#%% Fájl betöltése, előkészítése
# open gyógypélda
with open('gyogypelda_23k.txt', 'r') as f:
    seq = f.readlines()

seq = [line.replace('\n', '').split(' ')[:] for line in seq]
seq = seq[:-1] # üressor miatt
seq = [[float(item) for item in line] for line in seq] # float értékek
DS_EOS = min(seq[0]) # EOS a legkisebb elem (minden datasetben így van, ami txt)
DS_StOS = DS_EOS - 1 # StOS legyen StOS -1 értékű
DS_PAD = DS_EOS - 2 # Pad legyen EOS - 1 ->>> 0-lesz a pad
seq_max_len = max(len(row) for row in seq)
for idx, row in enumerate(seq):
    seq[idx] += [DS_PAD] *  (seq_max_len - len(row) + 1) # a max_len-t érje el az input, ezért paddeljük ki

seq = np.array(seq) # array
seq_input_c= np.append(seq, [DS_EOS, DS_PAD, DS_StOS]) 

#%% szótár elkészítése
seq_input_c=seq_target_c= np.unique(seq_input_c)
sequences = seq
input_characters, target_characters = seq_input_c, seq_target_c
max_length = seq_max_len
PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS

# input - target felosztása - felezéssel
input_seqs = []
target_seqs = []
classification_labels = []
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

# array
input_seqs = np.array(input_seqs)
target_seqs = np.array(target_seqs)

# ha konfidenciás vágás kell
if SPLIT_WITH_CONFIDENCES or CONFIDENCES_SHEET:
    input_seqs, target_seqs, classification_labels, conf_splits, confs = split_data_with_confidences()
else:
    conf_splits, confs = None, None
    
# tokenek számának és szekvenciák hosszának megállapítása
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(input_seq) for input_seq in input_seqs])
max_decoder_seq_length = max([len(target_seq) for target_seq in target_seqs])

print('Number of samples:', len(input_seqs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

#%% Modellek elkészítése
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%% Ha nem fér bele a memóriába a teljes fájl, daraboljuk fel
input_splits = list(range(100000,int(len(input_seqs)), 10000))
lower_input_splits = list(range(0,int(len(input_seqs))-5000,5000))

#%% X, Y tokenizáció
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = input_token_index

encoder_input_data = np.zeros(
    (len(input_seqs), max_encoder_seq_length),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_seqs), max_decoder_seq_length),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_seqs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

REV_SEQ = True
for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
    for t, char in enumerate(input_seq):
        if REV_SEQ == True:
            encoder_input_data[i, -(t+1)] = input_token_index[char]
        else:
            encoder_input_data[i, t] = input_token_index[char]
    for t, char in enumerate(target_seq):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# train, test split, majd shuffle
X_encoder_train, X_encoder_test, X_decoder_train, X_decoder_test, target_train, target_test = train_test_split(
        encoder_input_data, 
        decoder_input_data, 
        decoder_target_data, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
        )

#%% Model training
encoder_inputs = Input(shape=(None,), name='encoder_input')
x = Embedding(num_encoder_tokens, EMB_DIM, name='encoder_embedding')(encoder_inputs)
x, state_h, state_c = LSTM(LATENT_DIM,
                           return_state=True, name='encoder_lstm')(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,), name='decoder_input')
emb = Embedding(num_decoder_tokens, EMB_DIM, name='decoder_embedding')
x = emb(decoder_inputs)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense =  Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model.summary()
#plot_model(model, to_file='model.png', show_shapes =True)
# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!


model.fit([X_encoder_train, X_decoder_train], target_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2)


#model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_model_12_15_1422.h5')
# Save model
#model.save('seq2seq_alarm_3600epoch_Sas.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states, name='encoder_model')
encoder_model.summary()
#encoder_model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_encoder_12_15_1422.h5')
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
    [decoder_outputs] + decoder_states, name='decoder_model')
#plot_model(decoder_model, to_file='model_dec.png', show_shapes =True)
decoder_model.summary()

#%% Generálás
results_val1, results_val2, overall_results = [], [], []
#decoder_model.load_weights('kosar_decoder_w.h5')
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# transitions: in which we store the tree structure, which has [[parent], [child], [depth in tree]]

transitions, probabilities, states, full_sequences = [], [], [], []
analysed_seqs = np.random.randint(len(input_seqs), size=10)
for input_seq, encoder_input, true_seq in zip(input_seqs[analysed_seqs], encoder_input_data[analysed_seqs], target_seqs[analysed_seqs]):
    # Take one sequence (part of the training test)
    # for trying out decoding
    if PAD in list(true_seq):
        target_seq_length = list(true_seq).index(PAD)
    else:
        target_seq_length = len(true_seq)
    if PAD in list(input_seq):
        input_seq_length = list(input_seq).index(PAD)
    else:
        input_seq_length = list(input_seq)
    
    decoded_seq, _, depth, _ = decode_sequence(encoder_input.reshape(1,max_length))
    print('-')
    print('Input sequence: ', input_seq[:input_seq_length].astype(int))
    print('predicted sequence:  ', decoded_seq.astype(int))
    print('true sequence: ', true_seq[1:target_seq_length].astype(int))


    
    #transitions.append(seq_transitions)
    #probabilities.append(seq_probabilities)
    #states.append(state)
    #full_sequences.append(seqs)
    