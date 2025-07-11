import os  
os.environ['OMP_NUM_THREADS'] = '2'  
os.environ['MKL_NUM_THREADS'] = '2'  

import numpy as np  
import pandas as pd  
from tqdm import tqdm  
from collections import Counter  
import csv  
import math  
import multiprocessing as mp  
from itertools import product, combinations  
from multiprocessing import Pool  
from Bio import SeqIO  
import functools

from sklearn.model_selection import KFold  
from sklearn.preprocessing import StandardScaler  
from sklearn.utils import shuffle  
from sklearn.metrics import (  
    accuracy_score,   
    precision_score,   
    recall_score,   
    f1_score,   
    matthews_corrcoef,   
    confusion_matrix  
)  
 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier  
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LogisticRegression  
from xgboost import XGBClassifier  
from lightgbm import LGBMClassifier  
from catboost import CatBoostClassifier

def compute_dde(sequence, max_length=25):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    theo_freq = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            theo_freq[aa1+aa2] = padded_sequence.count(aa1) * padded_sequence.count(aa2) / (max_length ** 2)
    
    actual_freq = Counter([padded_sequence[i:i+2] for i in range(max_length-1)])
    
    dde = []
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dipeptide = aa1 + aa2
            if dipeptide in actual_freq:
                dde.append((actual_freq[dipeptide] - theo_freq[dipeptide]) / theo_freq[dipeptide])
            else:
                dde.append(-1)  # If the dipeptide doesn't appear, use -1
    
    return np.array(dde)

def compute_ctriad(sequence, max_length=25):
    groups = {
        0: 'AGV',
        1: 'ILFP',
        2: 'YMTS',
        3: 'HNQW',
        4: 'RK',
        5: 'DE',
        6: 'C'
    }
    
    def get_group(aa):
        for g, aas in groups.items():
            if aa in aas:
                return g
        return 7  # for 'X' or any other amino acid not in the groups
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    ctriad = [0] * 512  # 8^3 possible combinations (including the unknown group)
    for i in range(max_length - 2):
        g1 = get_group(padded_sequence[i])
        g2 = get_group(padded_sequence[i+1])
        g3 = get_group(padded_sequence[i+2])
        index = g1 * 64 + g2 * 8 + g3
        ctriad[index] += 1
    
    return np.array(ctriad) / (max_length - 2)

def compute_gaac(sequence, max_length=25):
    groups = {
        'G1': 'AVLIMC',
        'G2': 'FWYH',
        'G3': 'STNQ',
        'G4': 'KR',
        'G5': 'DE',
        'G6': 'GP'
    }
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    gaac = [0] * 6
    for aa in padded_sequence:
        for i, (_, aas) in enumerate(groups.items()):
            if aa in aas:
                gaac[i] += 1
                break
    
    return np.array(gaac) / max_length

def compute_z_scale(sequence, max_length=25):
    # Z-scale values for 20 standard amino acids
    z_scales = {
        'A': [-0.591, -1.302, -0.733],
        'R': [1.538, 1.502, -0.055],
        'N': [0.945, 0.828, 1.299],
        'D': [1.050, 0.302, -0.259],
        'C': [-1.343, 0.465, -0.862],
        'Q': [0.931, 1.169, -0.503],
        'E': [1.357, -1.453, 1.477],
        'G': [-0.384, 1.652, 1.330],
        'H': [0.336, -0.417, -1.673],
        'I': [-1.239, -0.547, 0.393],
        'L': [-1.019, -0.987, -0.663],
        'K': [1.831, -0.561, 0.533],
        'M': [-0.663, -1.524, 2.219],
        'F': [-1.006, -0.590, 1.891],
        'P': [0.189, 2.081, -1.628],
        'S': [-0.228, 1.399, -4.760],
        'T': [-0.032, 0.326, 2.213],
        'W': [-0.595, 0.009, 0.672],
        'Y': [0.260, 0.830, 3.097],
        'V': [-1.337, -0.279, -0.544]
    }
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    z_scale_features = []
    
    for aa in padded_sequence:
        if aa in z_scales:
            z_scale_features.extend(z_scales[aa])
        else:
            z_scale_features.extend([0, 0, 0])  # For 'X' or any other non-standard amino acid
    
    return np.array(z_scale_features)

def compute_dpc(sequence, max_length=25):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    dpc = []
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            count = sum(1 for i in range(max_length - 1) if 
                        padded_sequence[i] == aa1 and padded_sequence[i+1] == aa2)
            dpc.append(count / (max_length - 1))
    
    return np.array(dpc)

def compute_gdpc(sequence, max_length=25):
    groups = {
        'G1': 'AVLIMC',
        'G2': 'FWYH',
        'G3': 'STNQ',
        'G4': 'KR',
        'G5': 'DE',
        'G6': 'GP'
    }
    
    def get_group(aa):
        for g, aas in groups.items():
            if aa in aas:
                return g
        return 'G0'  # for 'X' or any other amino acid not in the groups
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    gdpc = []
    for g1 in groups.keys():
        for g2 in groups.keys():
            count = sum(1 for i in range(max_length - 1) if 
                        get_group(padded_sequence[i]) == g1 and get_group(padded_sequence[i+1]) == g2)
            gdpc.append(count / (max_length - 1))
    
    return np.array(gdpc)

def compute_binary(sequence, max_length=25):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    binary = []
    for aa in padded_sequence:
        encoding = [1 if aa == amino_acid else 0 for amino_acid in amino_acids]
        binary.extend(encoding)
    
    return np.array(binary)

def compute_ctdt(sequence, max_length=25):
    groups = {
        1: 'RKEDQN',
        2: 'GASTPHS',
        3: 'CVLIMFW',
        4: 'Y'
    }
    
    def get_group(aa):
        for g, aas in groups.items():
            if aa in aas:
                return g
        return 0  # for 'X' or any other amino acid not in the groups
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    transitions = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    
    ctdt = []
    for t1, t2 in transitions:
        count = sum(1 for i in range(max_length - 1) if 
                    (get_group(padded_sequence[i]) == t1 and get_group(padded_sequence[i+1]) == t2) or
                    (get_group(padded_sequence[i]) == t2 and get_group(padded_sequence[i+1]) == t1))
        ctdt.append(count / (max_length - 1))
    
    return np.array(ctdt)

def compute_ctdc(sequence, max_length=25):
    groups = {
        1: 'RKEDQN',
        2: 'GASTPHS',
        3: 'CVLIMFW',
        4: 'Y'
    }
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    composition = [0] * 4
    
    for aa in padded_sequence:
        for g, aas in groups.items():
            if aa in aas:
                composition[g-1] += 1
                break
    
    return np.array(composition) / max_length

def compute_cksaap(sequence, k=5):  
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  
    max_length = 25  
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')  
    
    cksaap = []  
    for aa1 in amino_acids:  
        for aa2 in amino_acids:  
            count = sum(1 for i in range(max_length - k) if   
                        padded_sequence[i] == aa1 and padded_sequence[i+k] == aa2)  
            cksaap.append(count / (max_length - k))  
    
    return np.array(cksaap)  

def compute_cksaagp(sequence, max_length=25, k=5):  
    groups = {  
        'G1': 'AVLIMC',  
        'G2': 'FWYH',  
        'G3': 'STNQ',  
        'G4': 'KR',  
        'G5': 'DE',  
        'G6': 'GP'  
    }  
    
    def get_group(aa):  
        for g, aas in groups.items():  
            if aa in aas:  
                return g  
        return 'G0'    
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')  
    
    cksaagp = []  
    for g1 in groups.keys():  
        for g2 in groups.keys():  
            count = sum(1 for i in range(max_length - k) if   
                        get_group(padded_sequence[i]) == g1 and get_group(padded_sequence[i+k]) == g2)  
            cksaagp.append(count / (max_length - k))  
    
    return np.array(cksaagp)  

def compute_paac(sequence, max_length=25, lambda_=5, w=0.1):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    aac = Counter(padded_sequence)
    paac = [aac[aa] / max_length for aa in amino_acids]
    
    # Hydrophobicity (H1) and hydrophilicity (H2) values for each amino acid
    H1 = {'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85, 'E': -0.74, 'G': 0.48,
          'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18,
          'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08}
    H2 = {'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0, 'Q': 0.2, 'E': 3.0, 'G': 0.0,
          'H': -0.5, 'I': -1.8, 'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3,
          'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5}
    
    def compute_correlation(prop):
        corr = []
        for i in range(1, lambda_ + 1):
            sum_prop = sum((prop.get(padded_sequence[j], 0) - prop.get(padded_sequence[j+i], 0)) ** 2 
                           for j in range(max_length - i))
            corr.append(sum_prop / (max_length - i))
        return corr
    
    h1_correlation = compute_correlation(H1)
    h2_correlation = compute_correlation(H2)
    
    paac.extend([w * (h1 + h2) for h1, h2 in zip(h1_correlation, h2_correlation)])
    return np.array(paac) / (1 + w * lambda_ * 2)

def compute_eaac(sequence, max_length=25, window_size=5):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    eaac = []
    for i in range(0, max_length - window_size + 1):
        window = padded_sequence[i:i+window_size]
        count = Counter(window)
        eaac.extend([count[aa] / window_size for aa in amino_acids])
    
    return np.array(eaac)

def compute_egaac(sequence, max_length=25, window_size=5):
    groups = {
        'G1': 'AVLIMC',
        'G2': 'FWYH',
        'G3': 'STNQ',
        'G4': 'KR',
        'G5': 'DE',
        'G6': 'GP'
    }
    
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    
    egaac = []
    for i in range(0, max_length - window_size + 1):
        window = padded_sequence[i:i+window_size]
        count = Counter(window)
        for group in groups.values():
            group_count = sum(count[aa] for aa in group)
            egaac.append(group_count / window_size)
    
    return np.array(egaac)

def compute_aac(sequence, max_length=25):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    aac = Counter(padded_sequence)
    return np.array([aac[aa] / max_length for aa in amino_acids])

def compute_apaac(sequence, max_length=25, lambda_=5, w=0.1):  
    sequence = sequence[:max_length].ljust(max_length, 'X')  
    
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  
    aac = compute_aac(sequence, max_length)  
    
    # Hydrophobicity (H), hydrophilicity (L), and side-chain mass (M) values for each amino acid  
    H = {'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85, 'E': -0.74, 'G': 0.48,  
         'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18,  
         'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08, 'X': 0}    
    L = {'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0, 'Q': 0.2, 'E': 3.0, 'G': 0.0,  
         'H': -0.5, 'I': -1.8, 'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3,  
         'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5, 'X': 0}    
    M = {'A': 15, 'R': 101, 'N': 58, 'D': 59, 'C': 47, 'Q': 72, 'E': 73, 'G': 1,  
         'H': 82, 'I': 57, 'L': 57, 'K': 73, 'M': 75, 'F': 91, 'P': 42, 'S': 31,  
         'T': 45, 'W': 130, 'Y': 107, 'V': 43, 'X': 0}    
    
    def compute_correlation(prop):  
        corr = []  
        for i in range(1, lambda_ + 1):  
            sum_prop = sum((prop[sequence[j]] - prop[sequence[j+i]]) ** 2   
                          for j in range(max_length - i))  
            corr.append(sum_prop / (max_length - i))  
        return corr  
    
    h_correlation = compute_correlation(H)  
    l_correlation = compute_correlation(L)  
    m_correlation = compute_correlation(M)  
    
    apaac = list(aac) + [w * (h + l + m) for h, l, m in zip(h_correlation, l_correlation, m_correlation)]  
    return np.array(apaac) / (1 + w * lambda_ * 3)

# BLOSUM62矩阵
blosum62_matrix = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
}

ctdd_groups = {
    'hydrophobicity': [
        ['R', 'K', 'E', 'D', 'Q', 'N'],
        ['G', 'A', 'S', 'T', 'P', 'H', 'Y'],
        ['C', 'L', 'V', 'I', 'M', 'F', 'W']
    ],
    'normalized_vdwv': [
        ['G', 'A', 'S', 'T', 'P', 'D', 'C'],
        ['N', 'V', 'E', 'Q', 'I', 'L'],
        ['M', 'H', 'K', 'F', 'R', 'Y', 'W']
    ],
    'polarity': [
        ['L', 'I', 'F', 'W', 'C', 'M', 'V', 'Y'],
        ['P', 'A', 'T', 'G', 'S'],
        ['H', 'Q', 'R', 'K', 'N', 'E', 'D']
    ],
    'polarizability': [
        ['G', 'A', 'S', 'D', 'T'],
        ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'],
        ['K', 'M', 'H', 'F', 'R', 'Y', 'W']
    ],
    'charge': [
        ['K', 'R'],
        ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
        ['D', 'E']
    ],
    'secondary_structure': [
        ['E', 'A', 'L', 'M', 'Q', 'K', 'R', 'H'],
        ['V', 'I', 'Y', 'C', 'W', 'F', 'T'],
        ['G', 'N', 'P', 'S', 'D']
    ],
    'solvent_accessibility': [
        ['A', 'L', 'F', 'C', 'G', 'I', 'V', 'W'],
        ['R', 'K', 'Q', 'E', 'N', 'D'],
        ['M', 'S', 'P', 'T', 'H', 'Y']
    ]
}

# GTPC groups
gtpc_groups = {
    'G1': 'IVLM',
    'G2': 'FYW',
    'G3': 'ASTNQ',
    'G4': 'RKH',
    'G5': 'DE',
    'G6': 'GP',
    'G7': 'C'
}

def compute_blosum62(sequence, max_length=25):
    padded_sequence = sequence[:max_length].ljust(max_length, 'X')
    feature_vector = []
    for aa in padded_sequence:
        if aa in blosum62_matrix:
            feature_vector.extend(blosum62_matrix[aa])
        else:
            feature_vector.extend([0] * 20)  
    return np.array(feature_vector)

def compute_ctdd(sequence):
    features = []
    for prop, groups in ctdd_groups.items():
        for i, group in enumerate(groups):
            count = sum(1 for aa in sequence if aa in group)
            features.append(count / len(sequence))
    return np.array(features)

def compute_tpc(sequence, max_length=25):  

    padded_sequence = sequence[:max_length].ljust(max_length, 'X')  
    
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  
    tripeptides = [''.join(i) for i in product(amino_acids, repeat=3)]  
    tpc_features = {tripeptide: 0 for tripeptide in tripeptides}  
    
    for i in range(len(padded_sequence) - 2):  
        tripeptide = padded_sequence[i:i+3]  
        if tripeptide in tpc_features:  
            tpc_features[tripeptide] += 1  
    
    total_count = sum(tpc_features.values())  
    if total_count == 0:  
        return np.zeros(len(tpc_features))  
    
    tpc_features = {k: v / total_count for k, v in tpc_features.items()}  
    return np.array(list(tpc_features.values()))  

def compute_gtpc(sequence, max_length=25):  

    padded_sequence = sequence[:max_length].ljust(max_length, 'X')  
    
    gtpc_features = {}  
    for g1 in gtpc_groups.keys():  
        for g2 in gtpc_groups.keys():  
            for g3 in gtpc_groups.keys():  
                gtpc_features[f'{g1}{g2}{g3}'] = 0  

    for i in range(len(padded_sequence) - 2):  
        try:  
            tripeptide = ''.join([  
                next(g for g, aa in gtpc_groups.items() if padded_sequence[i+j] in aa)  
                for j in range(3)  
            ])  
            gtpc_features[tripeptide] += 1  
        except StopIteration:  
            continue  

    total_count = sum(gtpc_features.values())  
    if total_count == 0:  
        return np.zeros(len(gtpc_features))  
    
    gtpc_features = {k: v / total_count for k, v in gtpc_features.items()}  
    return np.array(list(gtpc_features.values()))

def load_and_prepare_data(pos_file, neg_file):  

    pos_sequences = []  
    for record in SeqIO.parse(pos_file, "fasta"):  
        pos_sequences.append(str(record.seq))  
     
    neg_sequences = []  
    for record in SeqIO.parse(neg_file, "fasta"):  
        neg_sequences.append(str(record.seq))  
    
    pos_labels = [1] * len(pos_sequences)  
    neg_labels = [0] * len(neg_sequences)  
    
    all_sequences = pos_sequences + neg_sequences  
    all_labels = pos_labels + neg_labels  
    
    data = pd.DataFrame({  
        'Sequence': all_sequences,  
        'label': all_labels  
    })  
    
    data = shuffle(data, random_state=42)  
    
    return data  

def compute_features_for_model(sequence, model_name):  
    if model_name == 'rf':  
        return {  
            'blosum62': compute_blosum62(sequence),  
            'ctdd': compute_ctdd(sequence),  
            'dpc': compute_dpc(sequence),  
            'aac': compute_aac(sequence)  
        }  
    elif model_name == 'ert':  
        return {  
            'ctdd': compute_ctdd(sequence),  
            'eaac': compute_eaac(sequence),  
            'gaac': compute_gaac(sequence),  
            'paac': compute_paac(sequence),  
            'ctdc': compute_ctdc(sequence),  
            'apaac': compute_apaac(sequence),  
            'z_scale': compute_z_scale(sequence)  
        }  
    elif model_name == 'cab':  
        return {  
            'blosum62': compute_blosum62(sequence),  
            'aac': compute_aac(sequence),  
            'ctdc': compute_ctdc(sequence),  
            'ctdd': compute_ctdd(sequence),  
            'tpc': compute_tpc(sequence),  
            'dde': compute_dde(sequence),  
            'eaac': compute_eaac(sequence)  
        }  
    elif model_name == 'lgb':  
        return {  
            'ctdc': compute_ctdc(sequence),  
            'eaac': compute_eaac(sequence),  
            'gaac': compute_gaac(sequence),  
            'ctriad': compute_ctriad(sequence),  
            'egaac': compute_egaac(sequence),  
            'binary': compute_binary(sequence)  
        }  
    elif model_name == 'xgb':  
        return {  
            'aac': compute_aac(sequence),  
            'egaac': compute_egaac(sequence),  
            'ctdc': compute_ctdc(sequence),  
            'gdpc': compute_gdpc(sequence),  
            'blosum62': compute_blosum62(sequence),  
            'dpc': compute_dpc(sequence),  
            'dde': compute_dde(sequence)  
        }  
    elif model_name == 'svm':  
        return {  
            'blosum62': compute_blosum62(sequence),  
            'ctdd': compute_ctdd(sequence),  
            'aac': compute_aac(sequence),  
            'gtpc': compute_gtpc(sequence),  
            'gaac': compute_gaac(sequence)  
        }  
    elif model_name == 'knn':  
        return {  
            'ctdd': compute_ctdd(sequence),  
            'binary': compute_binary(sequence),  
            'ctriad': compute_ctriad(sequence),  
            'gtpc': compute_gtpc(sequence),  
            'apaac': compute_apaac(sequence),  
            'ctdc': compute_ctdc(sequence),  
            'cksaap': compute_cksaap(sequence, k=5),  
            'egaac': compute_egaac(sequence)  
        }  
    elif model_name == 'lr':  
        return {  
            'tpc': compute_tpc(sequence)  
        }  
    elif model_name == 'adb':  
        return {  
            'ctdc': compute_ctdc(sequence),  
            'aac': compute_aac(sequence),  
            'dpc': compute_dpc(sequence),  
            'binary': compute_binary(sequence),  
            'tpc': compute_tpc(sequence),  
            'cksaap': compute_cksaap(sequence, k=1)  
        }  

def get_model(model_name):  
    if model_name == 'rf':  
        return RandomForestClassifier(random_state=42, n_jobs=2)  
    elif model_name == 'ert':  
        return ExtraTreesClassifier(random_state=42, n_jobs=2)  
    elif model_name == 'cab':  
        return CatBoostClassifier(random_state=42, train_dir=None, logging_level='Silent', thread_count=2)   
    elif model_name == 'lgb':  
        return LGBMClassifier(random_state=42, n_jobs=2)  
    elif model_name == 'xgb':  
        return XGBClassifier(random_state=42, n_jobs=2)  
    elif model_name == 'svm':  
        return SVC(random_state=42, probability=True)    
    elif model_name == 'knn':  
        return KNeighborsClassifier(n_jobs=2)  
    elif model_name == 'lr':  
        return LogisticRegression(random_state=42, n_jobs=2)  
    elif model_name == 'adb':  
        return AdaBoostClassifier(random_state=42)  

def get_feature_count(model_name):  
    feature_counts = {  
        'rf': 439,  
        'ert': 307,  
        'cab': 213,  
        'lgb': 391,  
        'xgb': 105,  
        'svm': 247,  
        'knn': 422,  
        'lr': 386,  
        'adb': 133  
    }  
    return feature_counts[model_name]  

def evaluate_ensemble(models_dict, weights, X_test, y_test):  
    weighted_predictions = np.zeros((len(y_test),))  
    for model_name, (model, X) in models_dict.items():  

        pred_proba = model.predict_proba(X)[:, 1]  
        weighted_predictions += weights[model_name] * pred_proba  
    
    y_pred = (weighted_predictions >= 0.5).astype(int)  
    
    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred)  
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  
    specificity = tn / (tn + fp)  
    mcc = matthews_corrcoef(y_test, y_pred)  
    
    return accuracy, precision, recall, f1, specificity, mcc

def process_weight_combination(weight, model_names, model_features, y, kf):  
  
    weight_dict = dict(zip(model_names, weight))  
    fold_results = []  
    
    for train_index, test_index in kf.split(model_features['rf']):  
        models_dict = {}  
        for model_name in model_names:  
            X = model_features[model_name]  
            X_train, X_test = X[train_index], X[test_index]  
            y_train, y_test = y[train_index], y[test_index]  
            
            model = get_model(model_name)  
            model.fit(X_train, y_train)  
            models_dict[model_name] = (model, X_test)  
        
        fold_results.append(evaluate_ensemble(models_dict, weight_dict, X_test, y_test))  
    
    avg_results = np.mean(fold_results, axis=0)  
    return [*weight, *avg_results]  

def main():  
 
    pos_file = '/root/shared-nvme/jdle/TastePepAI/TasToxPred/dataset/pos_train.fasta'  
    neg_file = '/root/shared-nvme/jdle/TastePepAI/TasToxPred/dataset/neg_train.fasta'  
    test_file = '/root/shared-nvme/jdle/TastePepAI_for_users/singal_x1xx1/VAE_TastePeps_positive_Epoch_295.fasta'  
    
    print("Loading training data...")  
    data = load_and_prepare_data(pos_file, neg_file)  
    train_sequences = data['Sequence'].tolist()  
    y = data['label'].values  

    print("Loading test data...")  
    test_sequences = []  
    sequence_ids = []  
    for record in SeqIO.parse(test_file, "fasta"):  
        test_sequences.append(str(record.seq))  
        sequence_ids.append(record.id)  
    
    model_names = ['rf', 'ert', 'cab', 'lgb', 'xgb', 'svm', 'knn', 'lr', 'adb']  
    weights = {  
        'rf': 0.0, 'ert': 0.0, 'cab': 0.0, 'lgb': 0.0,  
        'xgb': 0.3, 'svm': 0.1, 'knn': 0.2, 'lr': 0.3, 'adb': 0.1  
    }  

    trained_models = {}  
    scalers = {}  
    feature_selectors = {}  
 
    for model_name in model_names:  
        print(f"\nProcessing {model_name}...")  

        print("Computing training features...")  
        train_features = {feature: [] for feature in compute_features_for_model(train_sequences[0], model_name).keys()}  
        for seq in tqdm(train_sequences):  
            seq_features = compute_features_for_model(seq, model_name)  
            for feature_name, feature_value in seq_features.items():  
                train_features[feature_name].append(feature_value)  
        
        X_train = np.hstack([np.array(train_features[feat]) for feat in train_features.keys()])  

        print("Scaling features...")  
        scaler = StandardScaler()  
        X_train_scaled = scaler.fit_transform(X_train)  
        scalers[model_name] = scaler  

        print("Selecting features...")  
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)  
        rf_selector.fit(X_train_scaled, y)  
        feature_importance = rf_selector.feature_importances_  
        feature_indices = np.argsort(feature_importance)[::-1]  
        n_features = get_feature_count(model_name)  
        selected_features = feature_indices[:n_features]  
        feature_selectors[model_name] = selected_features  
        
        X_train_selected = X_train_scaled[:, selected_features]  

        print("Training model...")  
        model = get_model(model_name)  
        model.fit(X_train_selected, y)  
        trained_models[model_name] = model  
 
    print("\nProcessing test sequences...")  
    predictions = np.zeros(len(test_sequences))  
    
    for model_name in model_names:  
        print(f"\nComputing predictions for {model_name}...")  

        test_features = {feature: [] for feature in compute_features_for_model(test_sequences[0], model_name).keys()}  
        for seq in tqdm(test_sequences):  
            seq_features = compute_features_for_model(seq, model_name)  
            for feature_name, feature_value in seq_features.items():  
                test_features[feature_name].append(feature_value)  
        
        X_test = np.hstack([np.array(test_features[feat]) for feat in test_features.keys()])  

        X_test_scaled = scalers[model_name].transform(X_test)  

        X_test_selected = X_test_scaled[:, feature_selectors[model_name]]  

        pred_proba = trained_models[model_name].predict_proba(X_test_selected)[:, 1]  
        predictions += weights[model_name] * pred_proba  

    results_df = pd.DataFrame({  
        'Sequence_ID': sequence_ids,  
        'Sequence': test_sequences,  
        'Prediction_Probability': predictions,  
        'Toxicity_Status': ['Toxic' if prob >= 0.5 else 'Non-toxic' for prob in predictions]    
    })  
    
    output_file = 'TasToxpred_test_predictions.csv'  
    results_df.to_csv(output_file, index=False)  
    print(f"\nResults saved to {output_file}")  

if __name__ == "__main__":  
    main()