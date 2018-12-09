#%% imports

import pandas as pd
import numpy as np
import math
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Params for dataset

VAC = '..\\dataset\\Sequences_alarm_warning_171208.xlsx'
KOSARAK = 'kosarak25k.txt'
FIFA = 'FIFA.txt'
BIBLE = 'BIBLE.txt'
BMS2 = 'BMS2.txt'
SZAMOLAS = 'gyogypelda_23k.txt'
VISSZA_1_2 = 'last_num_back_2.txt'
SZOVEG = 'szoveg.txt'
HALF_CUMSUM = 'half_cumsum.txt'

DATA = VAC

#%% Konfidenciaszámításhoz létező adatok

# confidences, if given
SHEET = '' # ha sheet='', akkor nem keresi excel munkalapon, különben sheet-en
CONFIDENCES_SHEET = None 

# temporal setting
WITHOUT_TEMPORAL = True # Ha az adatok temporállal nélkül vannak, false, egyébként true-val kiszedi  

# fault label for crossval
FAULT_LABELS_FILENAME = '' # Ha a szekvenciákhoz rendelhető valamilyen osztály, akkor ebben a fájlban keresse a címkéket

# sequence split settings
SEQUENCE_MIN_LENGTH = 3 # sequences with length under given min_length will be dropped
INPUT_LENGTH = None # if given, all sequences will have fixed input length
INPUT_WITH_FREQ = False
SPLIT_WITH_CONFIDENCES = False # számítson hozzá konfidenciát, és a magas konfidenciájú értékeknél vágjon egyet

REVERSE_SEQUENCES = True # a bemenő szekvencia iránya fordított legyen-e

#%% Paraméterek

TEST_SIZE = 0.0
RANDOM_STATE = 42
BATCH_SIZE = 1024
EPOCHS = 1024

EMB_DIM = 4
LATENT_DIM = 16


#%% Függvények
def bevitel(input_sequence):
    
    input_sequence = [target_token_index[s] for s in input_sequence]
    # PAD
    try:
        input_sequence += [target_token_index[PAD]] * (max_length-len(input_sequence))
    except:
        input_sequence[:max_length]
    input_sequence = np.array(input_sequence[::-1], dtype='f')
    input_sequence = input_sequence.reshape(1,max_length)
    
    return input_sequence


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


def decode_sequence(input_seq, threshold=0.1, early_stopping=True):
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
        tree = []
        token_list = [target_seq]
        states_list = [states_value]
        depth = 0
        prediction_depth = [depth]
        prediction_probabilities = [1.0]
        done = False
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        
        decoded_seq = []
        for i, token in enumerate(token_list):

            output_tokens, h, c = decoder_model.predict(
            [token[-1:]] + states_list[i])
        
            # Sample a token
            probability = np.max(output_tokens[0, -1, :])
            max_sampled_token_index = np.argmax(output_tokens[0, 0])
            """
            print(output_tokens[0,0].sum())
            print("")
            print(output_tokens[0,0])
            """
            sampled_token_indices = np.where(output_tokens[0, 0] > threshold)[0]
            
            max_sampled_char = reverse_target_char_index[max_sampled_token_index]
            sampled_chars = []
            [sampled_chars.append(reverse_target_char_index[new_token]) for new_token in sampled_token_indices]
            sequence = [reverse_target_char_index[item[0]] for item in token.tolist()]
            if depth == 0 and not done:
                decoded_seq.append(max_sampled_char)

            #elif target_token_index[decoded_seq[-1]] == token[-1:][0,0] and not done:
            elif decoded_seq == sequence[1:] and not done:
                    decoded_seq.append(max_sampled_char)

                
            if decoded_seq[-1] == EOS:
                done = True
            depth += 1
            # Update the target sequence (of length 1).
            
            for sampled_token in sampled_token_indices:
                
                if (token[-1:][0, 0] != target_token_index[EOS] and token[-1:][0, 0] != target_token_index[PAD] and (depth < max_decoder_seq_length or not early_stopping)):
                    
                    token_list.append(np.append(token, np.array([[sampled_token]]), axis=0))
                    # Update states
                    states_value = [h, c]
                    states_list.append(states_value)
                    prediction_depth.append(prediction_depth[i] + 1)
                    prediction_probabilities.append(output_tokens[0, -1, :][sampled_token])
                    sequence = [reverse_target_char_index[item[0]] for item in token.tolist()]
                    #tree.append([reverse_target_char_index[sampled_token], reverse_target_char_index[token[-1:][0,0]], prediction_depth[i] + 1, output_tokens[0, -1, :][sampled_token]])
                    tree.append([reverse_target_char_index[sampled_token], sequence, prediction_depth[i] + 1, output_tokens[0, -1, :][sampled_token]])
                # Exit condition: either hit max length
               
            # or find stop character.
            			
        decoded_seq = np.array(decoded_seq)
        
        return decoded_seq, token_list, prediction_depth, prediction_probabilities, tree


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


def dictionary_dist(input_sequence, plot=True):
    
    dictionary, bins = np.unique(input_sequence, return_counts=True)
    bins = bins[2:]
    dictionary = dictionary[2:]
    dictionary = dictionary[np.argsort(bins)]
    bins.sort()
    if plot:
    
        plt.figure(figsize=(50,50))
        plt.bar(np.arange(len(bins)), bins[::-1])
        plt.ylim([0, 1000])
        plt.yticks(np.arange(0,2000,20))
        plt.xticks(np.arange(0,len(bins), 100))
        plt.savefig('kosarak_dict_dist.png', dpi=300)
        plt.show()
        
    #return dict(zip(dictionary, bins))
    return dict(zip(dictionary, bins))

def cumulate_distrib(chars, bins, percentage=.9):
    
    sum_chars = bins.sum()
    cumper_chars = bins.cumsum()/sum_chars 
    indices = list(np.array(np.where(cumper_chars > percentage)[0], dtype=int))
    
    return np.array(chars)[indices], np.array(bins)[indices]
#%%KOSARAK, BIBLE, FIFA
if DATA is KOSARAK:

    with open(KOSARAK, 'r', encoding='utf-8') as f:
        KOSARAK_input_sequences = f.readlines()   
    KOSARAK_input_sequences = [line.replace('\n', '').split(' ')[::2] for line in KOSARAK_input_sequences]
    KOSARAK_input_sequences = filter(lambda line: len(line) < 14, KOSARAK_input_sequences)
    KOSARAK_input_sequences = [[float(item) for item in line] for line in KOSARAK_input_sequences]
    DS_EOS = min(KOSARAK_input_sequences[0])
    DS_StOS = DS_EOS - 1
    DS_PAD = DS_EOS - 2
    KOSARAK_input_max_len = max(len(row) for row in KOSARAK_input_sequences)
    for idx, row in enumerate(KOSARAK_input_sequences):
        KOSARAK_input_sequences[idx] += [DS_PAD] *  (KOSARAK_input_max_len - len(row) + 1)
        
    KOSARAK_input_sequences = np.array(KOSARAK_input_sequences)
    KOSARAK_input_characters= np.append(KOSARAK_input_sequences, [DS_EOS, DS_PAD, DS_StOS])
    KOSARAK_input_characters=KOSARAK_target_characters = np.unique(KOSARAK_input_characters)
    
    sequences = KOSARAK_input_sequences
    input_characters, target_characters = KOSARAK_input_characters, KOSARAK_target_characters
    max_length = KOSARAK_input_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS

if DATA is SZAMOLAS:
    
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
    seq_input_c = np.unique(seq_input_c)
    sequences = seq
    input_characters, target_characters = seq_input_c, seq_input_c
    max_length = seq_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS

if DATA is BIBLE:
    
    with open(BIBLE, 'r', encoding='utf-8') as f:
        BIBLE_input_sequences = f.readlines()
    BIBLE_input_sequences = [line.replace('\n', '') for line in BIBLE_input_sequences]
    BIBLE_input_sequences = [line.split(' ') for line in BIBLE_input_sequences]
    BIBLE_input_sequences = [line[::2] for line in BIBLE_input_sequences]
    BIBLE_input_sequences = [[float(item) for item in line] for line in BIBLE_input_sequences]
    DS_EOS = min(BIBLE_input_sequences[0])
    DS_StOS = DS_EOS - 1
    DS_PAD = DS_EOS - 2
    BIBLE_input_max_len = max(len(row) for row in BIBLE_input_sequences)
    for idx, row in enumerate(BIBLE_input_sequences):
        BIBLE_input_sequences[idx] += [DS_PAD] *  (BIBLE_input_max_len - len(row) + 1)
    
    BIBLE_input_sequences = np.array(BIBLE_input_sequences)
    BIBLE_input_characters = np.append(BIBLE_input_sequences, [DS_EOS, DS_PAD, DS_StOS])
    BIBLE_input_characters=BIBLE_target_characters = np.unique(BIBLE_input_characters)
    
    sequences = BIBLE_input_sequences
    input_characters, target_characters = BIBLE_input_characters, BIBLE_target_characters
    max_length = BIBLE_input_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS
        

if DATA is FIFA:
    
    with open(FIFA, 'r', encoding='utf-8') as f:
        FIFA_input_sequences = f.readlines()
    FIFA_input_sequences = [line.replace('\n', '') for line in FIFA_input_sequences]
    FIFA_input_sequences = [line.split(' ') for line in FIFA_input_sequences]
    FIFA_input_sequences = [line[::2] for line in FIFA_input_sequences]
    FIFA_input_sequences = [[float(item) for item in line] for line in FIFA_input_sequences]
    DS_EOS = min(FIFA_input_sequences[0])
    DS_StOS = DS_EOS - 1
    DS_PAD = DS_EOS - 2
    FIFA_input_max_len = max(len(row) for row in FIFA_input_sequences)
    for idx, row in enumerate(FIFA_input_sequences):
        FIFA_input_sequences[idx] += [DS_PAD] *  (FIFA_input_max_len - len(row) + 1)
    
    FIFA_input_sequences = np.array(FIFA_input_sequences)
    FIFA_input_characters = np.append(FIFA_input_sequences, [DS_EOS, DS_PAD, DS_StOS])
    FIFA_input_characters=FIFA_target_characters = np.unique(FIFA_input_characters)
    
    sequences = FIFA_input_sequences
    input_characters, target_characters = FIFA_input_characters, FIFA_target_characters
    max_length = FIFA_input_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS

if DATA is BMS2:
    
    with open(BMS2, 'r', encoding='utf-8') as f:
        BMS2_input_sequences = f.readlines()
    BMS2_input_sequences = [line.replace('\n', '') for line in BMS2_input_sequences]
    BMS2_input_sequences = [line.split(' ') for line in BMS2_input_sequences]
    BMS2_input_sequences = [line[::2] for line in BMS2_input_sequences]
    BMS2_input_sequences = [[float(item) for item in line] for line in BMS2_input_sequences]
    DS_EOS = min(BMS2_input_sequences[0])
    DS_StOS = DS_EOS - 1
    DS_PAD = DS_EOS - 2
    BMS2_input_max_len = max(len(row) for row in BMS2_input_sequences)
    for idx, row in enumerate(BMS2_input_sequences):
        BMS2_input_sequences[idx] += [DS_PAD] *  (BMS2_input_max_len - len(row) + 1)
        
    BMS2_input_sequences = np.array(BMS2_input_sequences)
    BMS2_input_characters = np.append(BMS2_input_sequences, [DS_EOS, DS_PAD, DS_StOS])
    BMS2_input_characters=BMS2_target_characters = np.unique(BMS2_input_characters)

    sequences = BMS2_input_sequences
    input_characters, target_characters = BMS2_input_characters, BMS2_target_characters
    max_length = BMS2_input_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS
    
if DATA is VAC:
    
    df_sequences = pd.read_excel(VAC, header=None, dtype=float)
    if WITHOUT_TEMPORAL:
        df_sequences = pd.DataFrame(df_sequences.as_matrix()[:, ::2])  # vagjuk ki a temporalokat, rakjuk vissza df-be
    #df_sequences = pd.read_excel(FILENAME, header=None, dtype=float, sheet_name=SHEET)
    EOS = df_sequences.values.min()  # end of sequences erteke a DF-ben a legkisebb érték (alapjáratod a PAD)
    PAD = EOS - 2  # padding erteke, PAD legyen a legkisebb, átkódolva 0 lesz (helyette jött az EOS, meg majd a StOS)
    StOS = EOS - 1
    
    input_characters = np.append(np.array(df_sequences), [EOS, PAD, StOS])
    input_characters = np.unique(input_characters)
    target_characters = input_characters
    df_sequences.replace(EOS, PAD, inplace=True)    # az eredtileg paddingelt szekvenciak pad erteket lecsokkentem, helyet csinalva a EOS-nak
    sequences = df_sequences.as_matrix()  # numpy array a DF-bol
    
    sequences[np.arange(sequences.shape[0]),np.argmin(sequences, axis=1)] = EOS
    df_sequences = pd.DataFrame(sequences)
    df_sequences[str(sequences.shape[1])] = PAD  # a leghosszabb szekvenciak vegere nem tudunk EOS-t rakni, ezert kell meg egy oszlop, alapertek = -6.0
    sequences = df_sequences.as_matrix()  # vissza numpy arraybe ... maceras!
    
    max_length = sequences.shape[1]

if DATA is VISSZA_1_2:
    
    with open(VISSZA_1_2, 'r') as f:
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
    seq_input_c = np.unique(seq_input_c)
    sequences = seq
    input_characters, target_characters = seq_input_c, seq_input_c
    max_length = seq_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS
    
if DATA is HALF_CUMSUM:
    
    with open(HALF_CUMSUM, 'r') as f:
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
    seq_input_c = np.unique(seq_input_c)
    sequences = seq
    input_characters, target_characters = seq_input_c, seq_input_c
    max_length = seq_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS

if DATA is SZOVEG:
    
    with open(SZOVEG, 'r') as f:
        seq = f.readlines()
    seq = [line.replace('\n', '').split(' ')[:] for line in seq]
    seq = seq[:-1] # üressor miatt
    #seq = [item for item in line] for line in seq] # float értékek
    DS_EOS = str(float(min(seq[0]))) # EOS a legkisebb elem (minden datasetben így van, ami txt)
    DS_StOS = str(float(DS_EOS) - 1) # StOS legyen StOS -1 értékű
    DS_PAD = str(float(DS_EOS) - 2)# Pad legyen EOS - 1 ->>> 0-lesz a pad
    seq_max_len = max(len(row) for row in seq)
    for idx, row in enumerate(seq):
        seq[idx] += [DS_PAD] *  (seq_max_len - len(row) + 1) # a max_len-t érje el az input, ezért paddeljük ki
    
    seq = np.array(seq) # array
    seq_input_c= np.append(seq, [DS_EOS, DS_PAD, DS_StOS]) 
    seq_input_c = np.unique(seq_input_c)
    sequences = seq
    input_characters, target_characters = seq_input_c, seq_input_c
    max_length = seq_max_len
    PAD, EOS, StOS = DS_PAD, DS_EOS, DS_StOS

#%% Split dataset into input and target

# input - target felosztása - felezéssel
input_seqs = []
target_seqs = []
classification_labels = []

if INPUT_WITH_FREQ:

    dist_dict = dictionary_dist(sequences, plot=False)
    chars, bins = np.array(list(dist_dict.keys())), np.array(list(dist_dict.values()))
    chars, new_bins = cumulate_distrib(chars, bins, percentage=0.37)
    chars = np.append(chars, [PAD, EOS, StOS])
    bins = np.append(bins, [1., 1., 1.])
    non_frequent_array = np.setdiff1d(input_characters, chars)
    
    frequent_sequences = np.isin(sequences, non_frequent_array, invert=True)
    input_characters, target_characters = chars, chars 
    
for row in tqdm(range(sequences.shape[0])):
    
    if INPUT_WITH_FREQ:
        sequence_length = list(sequences[row]).index(PAD)  # the end of the sequence
        if sequence_length > SEQUENCE_MIN_LENGTH:
            tmp_seq = sequences[row][frequent_sequences[row]]
            tmp_seq = np.append(tmp_seq, [PAD] * (len(frequent_sequences[row]) - frequent_sequences[row].sum()))
            sequences[row] = tmp_seq
            
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

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = input_token_index

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
decoder_dense = Dense(num_decoder_tokens, activation='sigmoid', name='dec_dense') # softmax
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#%%Split dataset into batches, fit model 

# input splits means the input batch size to load into the memory at one training batch
# lower input splits is to determine the starting index of the batch to load into the memory
input_splits = list(range(5000,int(len(input_seqs)), 5000))
lower_input_splits = list(range(0,int(len(input_seqs))-5000,5000))

# We do EPOCHS time training over the dataset, where we load input_split size batches for training
# only for one epoch, then load the next batch for one epoch. One epoch is indicated by epoch_id, where
# we do numerous memory load
# model.load_weights('base_05_11_1_50.h5')
if len(input_splits):
    for epoch_id, epoch in enumerate([1] * EPOCHS):
        print('epoch {}/{}'.format(epoch_id + 1, EPOCHS))
        for (lower_split, input_split) in zip(lower_input_splits, input_splits):
            print('\ttrain for: {}/{}'.format(input_split, len(input_seqs)))    
            encoder_input_data = np.zeros((5000, max_encoder_seq_length), dtype='float32')
            decoder_input_data = np.zeros((5000, max_decoder_seq_length), dtype='float32')
            decoder_target_data = np.zeros((5000, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
            
            for i, (input_seq, target_seq) in enumerate(zip(input_seqs[lower_split:input_split], target_seqs[lower_split:input_split])):
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
            encoder_input_data, 
            decoder_input_data, 
            decoder_target_data, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
            )       
            
            history = model.fit([X_encoder_train, X_decoder_train], target_train,
                      batch_size=BATCH_SIZE,
                      epochs=epoch,
                      verbose=0)
            
            
            
        print("accuracy: {}, loss: {}".format(history.history['acc'][0], history.history['loss'][0]))
else:
   
    encoder_input_data = np.zeros((len(input_seqs), max_encoder_seq_length), dtype='float32')
    decoder_input_data = np.zeros((len(input_seqs), max_decoder_seq_length), dtype='float32')
    decoder_target_data = np.zeros((len(input_seqs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    
    for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
        for t, char in enumerate(input_seq):
            if REVERSE_SEQUENCES == True:
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

    
    X_encoder_train, X_encoder_test, X_decoder_train, X_decoder_test, target_train, target_test = train_test_split(
        encoder_input_data, 
        decoder_input_data, 
        decoder_target_data, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
        )
    
    model.fit([X_encoder_train, X_decoder_train], target_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2)
#%% Define Encoder- and Decodermodel
# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states, name='encoder_model')
#encoder_model.load_weights('encoder_05_11_1_50.h5')
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
#decoder_model.load_weights('decoder_05_11_1_50.h5')

#%% Predict trees, sequences


encoder_input_data = np.zeros(
    (len(input_seqs), max_encoder_seq_length),
    dtype='float32')
for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
    for t, char in enumerate(input_seq):
        if REVERSE_SEQUENCES == True:
            encoder_input_data[i, -(t+1)] = input_token_index[char]
        else:
            encoder_input_data[i, t] = input_token_index[char]



results_val1, results_val2, overall_results = [], [], []



decoder_model.summary()
#decoder_model.load_weights('kosar_decoder_w.h5')
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
analysed_seqs = np.random.randint(len(input_seqs), size=10)
#analysed_seqs = np.arange(0,20)
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
    
    decoded_seq, token_list, depth, probabilities, _ = decode_sequence(encoder_input.reshape(1,max_length))
    print('-')
    try:
        print('Input sequence: ', input_seq[:input_seq_length].astype(int))
        print('predicted sequence:  ', decoded_seq.astype(int))
        print('true sequence: ', true_seq[1:target_seq_length].astype(int))
    except:
        print('Input sequence: ', input_seq[:input_seq_length])
        print('predicted sequence:  ', decoded_seq)
        print('true sequence: ', true_seq[1:target_seq_length])
best_tree, tree, tree_tokens, tree_probabilities, pytree = [], [], [], [], []
import evaluate
from ete3 import Tree, TreeStyle

i = 0
for input_seq, encoder_input in zip(input_seqs[analysed_seqs], encoder_input_data[analysed_seqs]):
    
    decoded_seq, tokens, _, probabilities, tree_objs = decode_sequence(encoder_input.reshape(1,max_length))
    tree_probabilities.append(probabilities)
    tree.append(tree_objs)
    _pytree, nodes = evaluate.build_tree(tree_objs, input_seq[:input_seq_length].astype(int), EOS=EOS, PAD=PAD, StOS=StOS, threshold=0.1)
    print(nodes)
    print(_pytree.get_ascii(attributes=["name", "prob"]))
    ts = TreeStyle()
    ts.show_leaf_name = True
    
    _pytree.render('tree{}.png'.format(i), tree_style=ts)
    
    i+=1
    pytree.append(_pytree)
    
    tree_tokens.append(tokens)
    best_tree.append(decoded_seq)
    

#%% BEVITEL
import evaluate
from ete3 import TreeStyle, TextFace, faces
input_s = [1,2,3,4]
#decoded_seq, token_list, depth, probabilities, tree_objs = decode_sequence(bevitel(input_s), threshold=0.1, early_stopping=False, EOS=EOS, PAD=PAD, StOS=StOS)
decoded_seq, token_list, depth, probabilities, tree_objs = decode_sequence(bevitel(input_s), threshold=0.1, early_stopping=True)

print('-')
try:
    print('Input sequence: ', input_s)
    print('predicted sequence:  ', decoded_seq.astype(int))
#    print('true sequence: ', true_seq[1:target_seq_length].astype(int))
except:
    print('Input sequence: ', input_s)
    print('predicted sequence:  ', decoded_seq)
#    print('true sequence: ', true_seq[1:target_seq_length])
_pytree, nodes = evaluate.build_tree(tree_objs, input_s, threshold=0.00)
print(_pytree.get_ascii(attributes=["name", "support"]))
ts = TreeStyle()
ts.scale = 240
ts.branch_vertical_margin = 5
ts.show_branch_length = False
ts.show_branch_support = True
ts.show_leaf_name = False
_pytree.show(tree_style=ts)
print('-')
try:
    print('Input sequence: ', input_s)
    print('predicted sequence:  ', decoded_seq.astype(int))
#    print('true sequence: ', true_seq[1:target_seq_length].astype(int))
except:
    print('Input sequence: ', input_s)
    print('predicted sequence:  ', decoded_seq)

#%%
"""
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
        sequence_length = list(target_seqs[seq_index]).index(PAD)
        true_seq = target_seqs[seq_index][1:sequence_length]
    else:
        sequence_length = list(target_seqs[seq_index]).index(EOS)
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

#%%
for fault in range(max(classification_labels) + 1):
    fault_indices = np.where(classification_labels == fault)[0]
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
            sequence_length = list(target_seqs_split[seq_index]).index(PAD)
            true_seq = target_seqs_split[seq_index][1:sequence_length]
        else:
            sequence_length = list(target_seqs_split[seq_index]).index(EOS)
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
"""
