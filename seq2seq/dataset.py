import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_sequences(filename, length_min_seq, temporal=True):
            """Load sequence dataset
            
            #Arguments:
                filename : the file, which stores the sequences - excel file, or .csv
                length_min_seq : the minimum sequence length in the dataset
                temporal : in case the dataset contains no temporal predicates, define it as false, otherwise it is inserted between each character
            
            #Returns:
                input_seqs : the input sequences of the data
                target_seqs : the target sequences
                input_characters : the character set (vocabulary) of the sequences
            """
            
            df_sequences = pd.read_excel(filename, header=None, dtype=float)
            EOS = df_sequences.values.min() # end of sequences erteke
            PAD = EOS - 2  # padding erteke, PAD legyen a legkisebb, átkódolva 0 lesz
            StOS = EOS - 1 # Start of Sequence jel
            input_characters = np.array(df_sequences)
            input_characters = np.append(input_characters, [EOS, PAD, StOS])
            input_characters = np.unique(input_characters)
            df_sequences.replace(EOS,PAD, inplace=True) # az eredtileg paddingelt szekvenciak pad erteket lecsokkentem, helyet csinalva a EOS-nak
            sequences = df_sequences.as_matrix() # numpy array a DF-bol
            if temporal:
                sequences = pd.DataFrame(sequences[:,::2]) # vagjuk ki a temporalokat, rakjuk vissza df-be
            sequences[str(sequences.shape[1])] = PAD # a leghosszabb szekvenciak vegere nem tudunk EOS-t rakni, ezert kell meg egy oszlop, alapertek = -6.0
            sequences = sequences.as_matrix() # vissza numpy arraybe ... maceras!
        
            input_seqs = []
            target_seqs = []
            for row in range(sequences.shape[0]):
                length_sequence = list(sequences[row]).index(PAD) # the end of the sequence
                if length_sequence > length_min_seq:
                    input_seq = sequences[row,:math.ceil(length_sequence/2)]
                    target_seq = sequences[row,math.ceil(length_sequence/2):]
                    target_seq_end = list(target_seq).index(PAD)
                    #target_seq[target_seq_end] = EOS
                    target_seq = np.insert(target_seq, 0, StOS)
                    input_seqs.append(input_seq)
                    target_seqs.append(target_seq)
            input_seqs = np.array(input_seqs)
            target_seqs = np.array(target_seqs)
            
            return input_seqs, target_seqs, sequences, input_characters, [PAD, EOS, StOS]
        
        
def load_faultclasses(filename):
            """Load fault classes of the sequences (if there is any)
            
            #Arguments:
                filename : the file, which stores the sequences - excel file, or .csv

            #Returns:
                faultclass : the true labels for each sequences
                y_label : the onehot encoded labels of each sequences
                --Encoders
                    YEncoder : 
            """        
    
            df_faultclass = pd.read_excel(filename, header=None, dtype=int)
            faultclass = df_faultclass.as_matrix()
            shape_faultclass = faultclass.shape
            YEncoder = preprocessing.LabelEncoder()
            YEncoder.fit(faultclass.flatten()) 
            Y_encoded = YEncoder.fit_transform(faultclass)
            y_label=Y_encoded
            faultclass=Y_encoded.reshape((shape_faultclass)) 
            YEncoder_OH = preprocessing.LabelBinarizer()
            YEncoder_OH.fit(faultclass)
            faultclass = YEncoder_OH.transform(faultclass)
            #faultclass = np_utils.to_categorical(faultclass)
            
            return faultclass, y_label, [YEncoder, YEncoder_OH]
        

def split_sequences_with_labels(input_seqs, target_seqs, sequences, length_min_seq, where_to_split, PAD, StOS, EOS, y_label):
    
            input_seqs = []
            target_seqs = []
            target_classes = []
            for row in range(sequences.shape[0]):
                current_faultclass = y_label[row]
                try:
                    length_sequence = list(sequences[row]).index(PAD) # the end of the sequence
                
                    
                    if length_sequence > length_min_seq:
                        input_seq = sequences[row,:length_sequence-where_to_split]
                        target_seq = sequences[row,length_sequence-where_to_split:]
                        target_seq_end = list(target_seq).index(PAD)
                        target_seq[target_seq_end] = EOS
                        target_seq = np.insert(target_seq, 0, StOS)
                        input_seqs.append(input_seq)
                        target_seqs.append(target_seq)
                        target_classes.append(current_faultclass)
                except:
                    print(row)
            input_seqs = np.array(input_seqs)
            target_seqs = np.array(target_seqs)
            faultclass = np.array(target_classes)
            return input_seqs, target_seqs, faultclass
        

def tokenize(input_seqs, target_seqs, input_characters, target_characters, reverse_sequence=True):
    
            num_encoder_tokens = len(input_characters)
            num_decoder_tokens = len(target_characters)
            max_encoder_seq_length = max([len(input_seq) for input_seq in input_seqs])
            max_decoder_seq_length = max([len(target_seq) for target_seq in target_seqs])
            
            print('Number of samples:', len(input_seqs))
            print('Number of unique input tokens:', num_encoder_tokens)
            print('Number of unique output tokens:', num_decoder_tokens)
            print('Max sequence length for inputs:', max_encoder_seq_length)
            print('Max sequence length for outputs:', max_decoder_seq_length)
            
            
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
            
            for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
                for t, char in enumerate(input_seq):
                    if reverse_sequence:
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
            
            return encoder_input_data, decoder_input_data, decoder_target_data, \
                [num_encoder_tokens, num_decoder_tokens], \
                [max_encoder_seq_length, max_decoder_seq_length], \
                [input_token_index, target_token_index]
                
                

