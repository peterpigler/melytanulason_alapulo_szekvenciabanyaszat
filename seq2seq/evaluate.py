import numpy as np
import pandas as pd
from ete3 import Tree, TreeStyle





def build_dictionary(tree_sequence, filename='one_tree.csv'):

    depths = np.array(tree_sequence, dtype=int)[:,2]
    transitions = np.array(tree_sequence,dtype=int)[:,:2]
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
    df.to_csv(filename)
    
    return dct

one_tree = seqs[8]
confidences = tree_conf[8]


def build_tree(one_tree, confidences):
    # one_tree - egy lista, amely a szekvenciasorozatokat tartalmazza, amiket lépésekben generál
    # confidences - a generálási lépéshez tartozó konfidencia
    
    confidences = np.insert(confidences, 0, 1.0)
    tree = Tree()
    nodes = {'': [1.0, tree]}
    for idx, seq in enumerate(one_tree):
        parent = ''
        for i in range(len(seq)): 
            if not str(seq[:i+1]) in tree:
                
                nodes[str(seq[:i+1])] = []
                nodes[str(seq[:i+1])].append(confidences[idx] * nodes[parent][0]) 
                nodes[str(seq[:i+1])].append(nodes[parent][1].add_child(name=str(seq), dist=nodes[str(seq[:i+1])][0]))

            else:
                parent = str(seq[:i+1])
    
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_length = True
    ts.show_branch_support = True
    print(tree)
    
    return tree


def find_frequency(one_tree, decoder_input_data):
    
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
        
    return occurances


print('finished..')