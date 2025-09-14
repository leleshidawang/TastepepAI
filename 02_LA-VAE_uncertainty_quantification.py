
import numpy as np  
from Bio import SeqIO  
import shutil  
import time
print("=" * 100)  
print("Welcome to LA-VAE ") 
print("\nIn this tool, each peptide in the training dataset is labeled with a sequence of five digits representing different tastes:")  
print("Position 1: Sour   (1=Reported Sour taste)")  
print("Position 2: Sweet  (1=Reported Sweet taste)")  
print("Position 3: Bitter (1=Reported Bitter taste)")  
print("Position 4: Salty  (1=Reported Salty taste)")  
print("Position 5: Umami  (1=Reported Umami taste)")  

print("\nExamples from training data:")  
print(">11xxx - A peptide reported to have both Sour and Sweet tastes")  
print(">x1xxx - A peptide reported to have Sweet taste")  
print(">xxx1x - A peptide reported to have Salty taste")  
print(">xxxx1 - A peptide reported to have Umami taste")  
print(">1xx11 - A peptide reported to have Sour, Salty and Umami tastes")  

print("\nWhen generating peptides, you have two options:")  
print("\n1. Single Pattern Mode:")  
print("   - Might provide more focused results for specific taste properties")  
print("   - Could be more suitable when targeting specific taste characteristics")  
print("   Example: >11xxx - Target peptides with Sour and Sweet tastes")  

print("\n2. Multiple Pattern Mode:")  
print("   - May help explore more sequence possibilities")  
print("   - Could potentially lead to more diverse peptide candidates")  
print("   Example: >x1xxx,xxx1x,xxxx1 - Explore peptides with different taste patterns")  

print("\nImportant Note About Pattern Input:")  
print("When entering your pattern(s), you can use:")  
print("- '1': indicates you want this taste")  
print("- 'x': indicates you don't have a preference for this taste")  
print("- '0': indicates you specifically don't want this taste")  

print("\nTraining Sample Selection:")  
print("- If your pattern(s) contain '0', the system will include both:")  
print("  * Positive samples: peptides matching your desired tastes")  
print("  * Negative samples: peptides that don't match your pattern")  
print("- If your pattern(s) only contain '1' and 'x', the system will use:")  
print("  * Only positive samples: peptides matching your desired tastes")  

print("\nNote: The actual results may vary depending on various factors")  
print("=" * 100)  

amino_acids = 'XACDEFGHIKLMNPQRSTVWY'  
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}  

sequences_found = False  

while not sequences_found:  
    use_multiple = input("Would you like to enter multiple taste patterns? (yes/no): ").lower()  
    
    if use_multiple == 'yes':  
        user_input = input("Enter your desired taste patterns separated by commas (e.g., x1xxx,xxx1x,xxxx1): ")  
        patterns = [pattern.strip() for pattern in user_input.split(',')]  
        
        valid_patterns = True  
        for pattern in patterns:  
            if len(pattern) != 5 or not all(char in '01x' for char in pattern):  
                print(f"Invalid pattern: {pattern}. Each pattern must be exactly five characters (0, 1, or x).")  
                valid_patterns = False  
                break  
        
        if not valid_patterns:  
            continue  
    else:  
        user_input = input("Enter your desired taste profile (e.g., xxxx1): ")  
        while len(user_input) != 5 or not all(char in '01x' for char in user_input):  
            print("Invalid input. Please ensure you enter exactly five characters, each either 0, 1, or x.")  
            user_input = input("Enter your desired taste peptide (e.g., xxxx1): ")  
        patterns = [user_input]  

    use_negative_samples = any('0' in pattern for pattern in patterns)  

    sequences = []  
    sequences_not_matched = []  

    for record in SeqIO.parse("Input_taste_peptides.fasta", "fasta"):  

        is_matched = False  
        for pattern in patterns:  
            match = True  
            for i, char in enumerate(pattern):  
                if char == '1' and record.id[i] != '1':  
                    match = False  
                    break  
                elif char == '0' and record.id[i] == '1':  
                    match = False  
                    break  
            if match:  
                is_matched = True  
                break  
                
        if is_matched:  
            sequences.append(str(record.seq))  
        elif use_negative_samples:   
            sequences_not_matched.append(str(record.seq))  

    if not sequences:  
        print("No peptides found matching any of the specified patterns. Please try again.")  
    else:  
        sequences_found = True  
        if len(patterns) > 1:  
            print(f"Found {len(sequences)} peptides matching the combined patterns: {', '.join(patterns)}")  
        else:  
            print(f"Found {len(sequences)} peptides matching the requirement: {patterns[0]}")  
        
        if use_negative_samples:  
            print(f"Found {len(sequences_not_matched)} peptides not matching the requirement(s).")  
            print("These will be used as negative samples since your patterns contain '0'.")  
        else:  
            print("Your patterns only contain '1' and 'x', so only positive samples will be used for training.")  
            sequences_not_matched = []   

        sequences_encoded = []  
        for seq in sequences:  
            seq_padded = seq.ljust(14, 'X')  
            sequences_encoded.append([aa_to_int[aa] for aa in seq_padded])  

        sequences_encoded_onehot = np.zeros((len(sequences_encoded), 14, 21), dtype=int)  
        for i, sequence in enumerate(sequences_encoded):  
            for j, aa in enumerate(sequence):  
                sequences_encoded_onehot[i, j, aa] = 1  

        if use_negative_samples and sequences_not_matched:  
            sequences_not_matched_encoded = []  
            for seq in sequences_not_matched:  
                seq_padded = seq.ljust(14, 'X')  
                sequences_not_matched_encoded.append([aa_to_int[aa] for aa in seq_padded])  

            sequences_not_matched_encoded_onehot = np.zeros((len(sequences_not_matched_encoded), 14, 21), dtype=int)  
            for i, sequence in enumerate(sequences_not_matched_encoded):  
                for j, aa in enumerate(sequence):  
                    sequences_not_matched_encoded_onehot[i, j, aa] = 1

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, LSTM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import heapq
import glob
from scipy.spatial import distance
import pandas as pd
from Bio import Align  
from tqdm import tqdm  
import networkx as nx  
from multiprocessing import Pool  
import itertools  

learning_rate = 0.001
batch_size = 16
epochs = 500
dropout_rate = 0.0
l1_reg = 0.01
new_sequences_num = 10000
extend_epochs_num = epochs
original_dim = (14, 21)
intermediate_dim = 200
latent_dim = 2000
conv1d_filters = 32

optimizer = Adam(learning_rate=learning_rate)

original_inputs = tf.keras.Input(shape=original_dim, name='encoder_input')
x = Conv1D(conv1d_filters, 3, activation="relu", padding="same")(original_inputs)
x = Dropout(dropout_rate)(x)  
x = Flatten()(x)
z_mean = Dense(latent_dim, name='z_mean', kernel_regularizer=regularizers.l1(l1_reg))(x)  
z_log_var = Dense(latent_dim, name='z_log_var', kernel_regularizer=regularizers.l1(l1_reg))(x)  
encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean, z_log_var], name='encoder')

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = Dense(conv1d_filters * 14, activation='relu', kernel_regularizer=regularizers.l1(l1_reg))(latent_inputs)  
x = Reshape((14, conv1d_filters))(x)
x = Conv1D(conv1d_filters, 3, activation="relu", padding="same")(x)
x = Dropout(dropout_rate)(x)  
x = Flatten()(x)
outputs = Dense(original_dim[0] * original_dim[1], kernel_regularizer=regularizers.l1(l1_reg))(x)
outputs = Reshape((original_dim[0], original_dim[1]))(outputs)
outputs = Activation('softmax')(outputs)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = Sampling()([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= original_dim[0] * original_dim[1]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

use_negative_samples = any('0' in pattern for pattern in patterns) 

vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizer)

class CustomCallback(Callback):  
    def __init__(self, sequences, dataset_name, num_epochs=epochs, num_best=10, extend_epochs=extend_epochs_num):  
        super().__init__()  
        self.sequences = sequences  
        self.dataset_name = dataset_name  
        self.num_epochs = num_epochs  
        self.num_best = num_best  
        self.extend_epochs = extend_epochs   
        self.is_extended = False    
        self.best_loss_first_half = None  
        self.best_loss_first_half_epoch = None  
        self.better_losses_second_half = []  
        self.int_to_aa = {i: aa for aa, i in aa_to_int.items()}

    def calculate_vae_confidence(self, sequences):
        """Calculate confidence based on VAE uncertainty"""
        z_mean, z_log_var = encoder.predict(sequences, verbose=0)

        uncertainty = np.mean(np.exp(z_log_var), axis=1)

        confidence = 1 / (1 + np.exp(uncertainty - 5.0))  
        
        return confidence

    def on_epoch_end(self, epoch, logs=None):  
        total_loss = logs.get('loss')  
        reconstruction_loss = logs.get('reconstruction_loss')  
        kl_loss = logs.get('kl_loss')  

        print(f'Epoch: {epoch+1}, Loss: {total_loss}, Reconstruction Loss: {reconstruction_loss}, KL Loss: {kl_loss}')  

        if epoch < self.num_epochs // 2:  
            if self.best_loss_first_half is None or total_loss < self.best_loss_first_half[0]:  
                self.best_loss_first_half = (total_loss, reconstruction_loss, kl_loss)  
                self.best_loss_first_half_epoch = epoch  
        elif total_loss < self.best_loss_first_half[0] and reconstruction_loss < self.best_loss_first_half[1] and kl_loss < self.best_loss_first_half[2]:  
            self.better_losses_second_half.append((total_loss, epoch, reconstruction_loss, kl_loss))  
        
            self.save_new_sequences_and_plot(epoch)  
            self.model.stop_training = True  
         
        if use_negative_samples:  
            negative_filename = f'z_mean_negative_Epoch_{epoch+1}.npy'  
            generated_negative_filename = f'z_mean_generated_negative_Epoch_{epoch+1}.npy'  
            if os.path.exists(negative_filename) and os.path.exists(generated_negative_filename):  
                self.plot_saved_four_latent_spaces(epoch)  
        else:  
            self.plot_saved_four_latent_spaces(epoch)  

    def on_train_end(self, logs=None):  
          
        if self.is_extended:  
            print("\nExtended training completed.")  
            if self.better_losses_second_half:  
                print("Better losses found during extended training:")  
                for loss in self.better_losses_second_half:  
                    print(f"Epoch: {loss[1]+1}, Total Loss: {loss[0]:.6f}, Reconstruction Loss: {loss[2]:.6f}, KL Loss: {loss[3]:.6f}")  
            else:  
                print("No better losses found during extended training.")  
            return  

        if not self.better_losses_second_half and not self.is_extended:  
            print("\nNo better loss found in the first training phase, extending training...")  
            self.is_extended = True  
            self.better_losses_second_half = []  
            self.model.fit(self.sequences, epochs=self.extend_epochs, verbose=0, callbacks=[self])  
        else:  
            print(f"\nBest loss in the first half (Epoch {self.best_loss_first_half_epoch+1}):")  
            print(f"Total Loss: {self.best_loss_first_half[0]:.6f}, Reconstruction Loss: {self.best_loss_first_half[1]:.6f}, KL Loss: {self.best_loss_first_half[2]:.6f}\n")  
        
            if self.better_losses_second_half:  
                print("Better losses found:")  
                for loss in self.better_losses_second_half:  
                    print(f"Epoch: {loss[1]+1}, Total Loss: {loss[0]:.6f}, Reconstruction Loss: {loss[2]:.6f}, KL Loss: {loss[3]:.6f}")  
            else:  
                print("No better losses found in the second half.")

    def save_new_sequences_and_plot(self, epoch):  
        n = new_sequences_num  
        z_samples = np.random.normal(size=(n, latent_dim))  
        x_decoded = decoder.predict(z_samples, verbose=0)

        vae_confidence = self.calculate_vae_confidence(x_decoded)
        
        sequences_decoded = []  
        for i in range(n):  
            sequence_decoded = []  
            for j in range(original_dim[0]):  
                aa = self.int_to_aa[np.argmax(x_decoded[i, j, :])]  
                sequence_decoded.append(aa)  
            sequences_decoded.append(''.join(sequence_decoded))  

        with open(f'VAE_TastePeps_{self.dataset_name}_Epoch_{epoch+1}.txt', 'w') as file:  
            for idx, (seq, conf) in enumerate(zip(sequences_decoded, vae_confidence)):
                file.write(f"Index: {idx}, Sequence: {seq}, VAE_Confidence: {conf:.6f}\n")

        z_mean, _ = encoder.predict(self.sequences, verbose=0)  
        z_mean_generated, _ = encoder.predict(x_decoded, verbose=0)  
        np.save(f'z_mean_{self.dataset_name}_Epoch_{epoch+1}.npy', z_mean)  
        np.save(f'z_mean_generated_{self.dataset_name}_Epoch_{epoch+1}.npy', z_mean_generated)  

        print(f"New sequences for epoch {epoch+1} have been saved with VAE confidence scores.")  

        self.plot_saved_four_latent_spaces(epoch)  

    def plot_saved_four_latent_spaces(self, epoch):  
        patterns = [  
            'z_mean_positive_Epoch_*.npy',  
            'z_mean_generated_positive_Epoch_*.npy',  
        ]  

        try:  
            z_mean_positive = np.load(glob.glob(patterns[0])[0])  
            z_mean_generated_positive = np.load(glob.glob(patterns[1])[0])  

            all_data = [z_mean_positive, z_mean_generated_positive]  

            if use_negative_samples:  
                z_mean_negative = np.load(glob.glob('z_mean_negative_Epoch_*.npy')[0])  
                z_mean_generated_negative = np.load(glob.glob('z_mean_generated_negative_Epoch_*.npy')[0])  
                all_data.extend([z_mean_negative, z_mean_generated_negative])  

        except IndexError:  
            return  

        try:  
            pca = PCA(n_components=2)  
            pca.fit(np.concatenate(all_data))  

            z_mean_positive_pca = pca.transform(z_mean_positive)  
            z_mean_generated_positive_pca = pca.transform(z_mean_generated_positive)  

            plt.rcParams.update({'font.size': 24})  
            plt.figure(figsize=(16, 12))  

            plt.scatter(z_mean_positive_pca[:, 0], z_mean_positive_pca[:, 1],   
                       color='purple', alpha=0.3, label='Positive training data', s=100)  
            plt.scatter(z_mean_generated_positive_pca[:, 0], z_mean_generated_positive_pca[:, 1],   
                       color='yellow', alpha=0.7, label='Generated positive data', s=100)  

            if use_negative_samples:  
                z_mean_negative_pca = pca.transform(z_mean_negative)  
                plt.scatter(z_mean_negative_pca[:, 0], z_mean_negative_pca[:, 1],   
                           color='green', alpha=0.3, label='Negative training data', s=100)  

            plt.xlabel('PC 1')  
            plt.ylabel('PC 2')  
            plt.legend()  
            plt.savefig(f'Latent_space_comparison_Epoch_{epoch+1}.png', dpi=600)  
            plt.close()  

            print(f"Latent space plot for saved data of epoch {epoch+1} has been generated.")  

            self.calculate_distances()  

        except Exception as e:  
            print(f"Error in plot generation: {str(e)}")  
            import traceback  
            traceback.print_exc()

    def calculate_distances(self):  
        try:  
            z_mean_positive = np.load(glob.glob('z_mean_positive_Epoch_*.npy')[0])  
            z_mean_generated_positive = np.load(glob.glob('z_mean_generated_positive_Epoch_*.npy')[0])  

            n = new_sequences_num
            z_samples_for_conf = np.random.normal(size=(n, latent_dim))
            x_decoded_for_conf = decoder.predict(z_samples_for_conf, verbose=0)
            vae_conf = self.calculate_vae_confidence(x_decoded_for_conf)
            
            data = []  
            all_dist_to_positive_data = []  
    
            for i, z in enumerate(z_mean_generated_positive):   

                dist_to_positive = distance.cdist([z], z_mean_positive, 'euclidean')[0]  
                dist_to_positive.sort()  

                avg_dist_to_positive = np.mean(dist_to_positive[:5])  
                std_dev_positive = np.std(dist_to_positive[:5])  
            
                data_entry = {  
                    'index': i,  
                    'avg_dist_to_positive': avg_dist_to_positive,  
                    'std_dev_positive': std_dev_positive,
                    'vae_confidence': vae_conf[i],
                }  
  
                if use_negative_samples:  
                    try:  
                        z_mean_negative = np.load(glob.glob('z_mean_negative_Epoch_*.npy')[0])  
                        dist_to_negative = distance.cdist([z], z_mean_negative, 'euclidean')[0]  
                        dist_to_negative.sort()  

                        avg_dist_to_negative = np.mean(dist_to_negative[:5])  
                        std_dev_negative = np.std(dist_to_negative[:5])   
                    
                        diff_avg_dist_pos_neg = avg_dist_to_positive - avg_dist_to_negative  

                        data_entry.update({  
                            'avg_dist_to_negative': avg_dist_to_negative,  
                            'std_dev_negative': std_dev_negative,  
                            'diff_avg_dist_pos_neg': diff_avg_dist_pos_neg,  
                        })  
                    
                        all_dist_to_negative_data = []  
                        all_dist_to_negative_data.append({'index': i, 'distances': dist_to_negative.tolist()})  

                    except Exception as e:  
                        print(f"Error processing negative samples: {str(e)}")  

                data.append(data_entry)  
                all_dist_to_positive_data.append({'index': i, 'distances': dist_to_positive.tolist()})  
            
            df = pd.DataFrame(data)  
            if use_negative_samples:  
                df.sort_values(by=['avg_dist_to_positive', 'avg_dist_to_negative'],   
                             ascending=[True, False], inplace=True)  
            else:  
                df.sort_values(by='avg_dist_to_positive', ascending=True, inplace=True)  
            
            df.to_csv('distances.csv', index=False)  

            df_all_dist_to_positive = pd.DataFrame(all_dist_to_positive_data)  
            df_all_dist_to_positive.to_csv('all_distances_to_positive.csv', index=False)  

            if use_negative_samples and 'all_dist_to_negative_data' in locals():  
                df_all_dist_to_negative = pd.DataFrame(all_dist_to_negative_data)  
                df_all_dist_to_negative.to_csv('all_distances_to_negative.csv', index=False)  

            print("Distance calculations completed with VAE confidence scores.")

        except Exception as e:  
            print(f"Error in calculate_distances: {str(e)}")

    @staticmethod  
    def calculate_similarity_nw(seq1, seq2):  
        aligner = Align.PairwiseAligner()  
        aligner.mode = 'global'  
        aligner.match_score = 2.0  
        aligner.mismatch_score = -1.0  
        aligner.open_gap_score = -0.5  
        aligner.extend_gap_score = -0.1  
        
        score = aligner.score(seq1, seq2)  
        max_length = max(len(seq1), len(seq2))  
        similarity = score / (2 * max_length) * 100  
        return similarity  
    
    @staticmethod  
    def calculate_similarities_parallel(args):  
        seq1, seq2 = args  
        similarity = CustomCallback.calculate_similarity_nw(seq1['sequence'], seq2['sequence'])  
        if similarity >= 70:  
            return (seq1['index'], seq2['index'], similarity)  
        return None  
    
    @staticmethod  
    def create_similarity_network(sequences):  
        print("Constructing sequence similarity network...")  
        G = nx.Graph()  
        
        for seq in sequences:  
            G.add_node(seq['index'], sequence=seq['sequence'])  
        
        seq_pairs = list(itertools.combinations(sequences, 2))  
        
        with Pool(processes=10) as pool:  
            similarity_results = list(tqdm(  
                pool.imap(CustomCallback.calculate_similarities_parallel, seq_pairs),  
                total=len(seq_pairs),  
                desc="Calculate sequence similarity"  
            ))  
        
        edge_count = 0  
        for result in similarity_results:  
            if result is not None:  
                id1, id2, similarity = result  
                G.add_edge(id1, id2, weight=similarity)  
                edge_count += 1  
        
        print(f"Network construction completed: {len(sequences)} nodes and {edge_count} edges")  
        return G  
    
    @staticmethod  
    def select_cluster_representative(cluster, sequences):  
        if len(cluster) == 1:  
            seq_index = list(cluster)[0]  
            return next(seq for seq in sequences if seq['index'] == seq_index)  
        
        best_score = float('-inf')  
        representative = None  
        
        for seq1_index in cluster:  
            seq1 = next(seq for seq in sequences if seq['index'] == seq1_index)  
            total_similarity = 0  
            for seq2_index in cluster:  
                if seq1_index != seq2_index:  
                    seq2 = next(seq for seq in sequences if seq['index'] == seq2_index)  
                    similarity = CustomCallback.calculate_similarity_nw(seq1['sequence'], seq2['sequence'])  
                    total_similarity += similarity  
            avg_similarity = total_similarity / (len(cluster) - 1)  
            
            if avg_similarity > best_score:  
                best_score = avg_similarity  
                representative = seq1  
        
        return representative  
    
    @staticmethod  
    def organize_output_files():  
        timestamp = time.strftime("%Y%m%d_%H%M%S")  
        result_dir = f"results_{timestamp}"  
        if not os.path.exists(result_dir):  
            os.makedirs(result_dir)  
    
        try:
            txt_files = glob.glob('VAE_TastePeps_positive_Epoch_*.txt')  
            if not txt_files and not os.path.exists('distances.csv'):  
                print("\nTraining Status: No Improvement")  
                print("No better model found during extended training.")  
                print("Consider reinitializing the model.")  
                return   

            for file in glob.glob('VAE_TastePeps_positive_Epoch_*.txt'):  
                shutil.move(file, os.path.join(result_dir, file))  
    
            for file in glob.glob('VAE_TastePeps_negative_Epoch_*.txt'):  
                os.remove(file)  
    
            for file in glob.glob('Latent_space_comparison_Epoch_*.png'):  
                os.remove(file)  
    
            if os.path.exists('distances.csv'):  
                shutil.move('distances.csv', os.path.join(result_dir, 'distances.csv'))  
    
            for file in glob.glob('*.npy'):  
                os.remove(file)  
    
            for file in ['all_distances_to_positive.csv', 'all_distances_to_negative.csv']:  
                if os.path.exists(file):  
                    os.remove(file)  
    
            df = pd.read_csv(os.path.join(result_dir, 'distances.csv'))  
            
            if 'diff_avg_dist_pos_neg' in df.columns:  
                df = df.nsmallest(n=int(len(df) * 0.25), columns=['diff_avg_dist_pos_neg'])  
            else:  
                df = df.nsmallest(n=int(len(df) * 0.25), columns=['avg_dist_to_positive'])  
            
            selected_indices = df['index'].tolist()  
            
            seq_file = glob.glob(os.path.join(result_dir, 'VAE_TastePeps_positive_Epoch_*.txt'))[0]  
            epoch_num = seq_file.split('Epoch_')[-1].split('.')[0]  
            
            sequences = []  
            with open(seq_file, 'r') as f:  
                for line in f:  
                    if line.strip():  
                        parts = line.strip().split(', ')
                        idx = int(parts[0].split(':')[1].strip())  
                        if idx in selected_indices:  
                            seq = parts[1].split(':')[1].strip()  

                            vae_conf = 0.0
                            if len(parts) > 2 and 'VAE_Confidence:' in parts[2]:
                                vae_conf = float(parts[2].split(':')[1].strip())
                            sequences.append({  
                                'index': idx,  
                                'sequence': seq.replace('X', ''),
                                'vae_confidence': vae_conf
                            })  
    
            G = CustomCallback.create_similarity_network(sequences)  
            clusters = list(nx.connected_components(G))  
            
            representative_sequences = []  
            for cluster in clusters:  
                representative = CustomCallback.select_cluster_representative(cluster, sequences)  
                if representative:  
                    representative_sequences.append(representative)  
    
            fasta_path = os.path.join(result_dir, f'VAE_TastePeps_positive_Epoch_{epoch_num}.fasta')  
            with open(fasta_path, 'w') as f:  
                for seq in representative_sequences:  
                    confidence_info = f"VAE_conf:{seq.get('vae_confidence', 0.0):.6f}"
                    f.write(f">index_{seq['index']}_{confidence_info}\n{seq['sequence']}\n")  
    
            os.remove(seq_file)  
    
            print(f"Files have been organized in directory: {result_dir}")  
            print(f"Selected and clustered sequences have been saved to {fasta_path}")  
            print(f"Initial sequences: {len(sequences)}")  
            print(f"Final representative sequences after clustering: {len(representative_sequences)}")  
            print("All sequences now include VAE confidence scores in the output.")
    
        except Exception as e:  
            print(f"Error organizing files: {str(e)}")  
            import traceback  
            traceback.print_exc()
    
custom_cb_positive = CustomCallback(sequences=sequences_encoded_onehot, dataset_name='positive')  
vae.fit(sequences_encoded_onehot, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[custom_cb_positive])  

if use_negative_samples and len(sequences_not_matched_encoded_onehot) > 0:  
    custom_cb_negative = CustomCallback(sequences=sequences_not_matched_encoded_onehot, dataset_name='negative')  
    vae.fit(sequences_not_matched_encoded_onehot, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[custom_cb_negative])  

CustomCallback.organize_output_files()
