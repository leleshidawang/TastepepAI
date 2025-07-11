import os
from Bio import SeqIO  
from Bio.SeqUtils.ProtParam import ProteinAnalysis  
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor  
import pandas as pd  
import numpy as np  
import warnings  
warnings.filterwarnings('ignore')  

def calculate_sequence_properties(sequence):  
    try:  
        sequence_str = str(sequence)  
        protein = ProteinAnalysis(sequence_str)  
        
        global_desc = GlobalDescriptor(sequence_str)  
        global_desc.calculate_all()  
        
        pep_desc = PeptideDescriptor([sequence_str], 'eisenberg')  
        try:  
            pep_desc.calculate_moment()  
            hydrophobic_moment = pep_desc.descriptor[0][0]  
        except (IndexError, AttributeError):  
            hydrophobic_moment = None  
            
        sol_desc = PeptideDescriptor([sequence_str], 'pepcats')  
        try:  
            sol_desc.calculate_global()  
            solubility_features = sol_desc.descriptor[0]  
        except (IndexError, AttributeError):  
            solubility_features = [None] * 3  
            
        properties = {  
            'GRAVY': protein.gravy(),  
            'Isoelectric_Point': protein.isoelectric_point(),  
            'Charge_at_pH7': protein.charge_at_pH(7.0),  
            'Molecular_Weight': protein.molecular_weight(),  
            'Aromaticity': protein.aromaticity(),  
            'Instability_Index': protein.instability_index(),  
            'Secondary_Structure': protein.secondary_structure_fraction(),  
            'Molar_Extinction': protein.molar_extinction_coefficient(),  
            
            'Aliphatic_Index': global_desc.descriptor[0][0],  
            'Charge_Density': global_desc.descriptor[0][2],  
            'Hydrophobic_Ratio': global_desc.descriptor[0][4],  
            'Hydrophobic_Moment': hydrophobic_moment,  
            
            'Solubility_Score': solubility_features[0] if solubility_features[0] is not None else None,  
            'Hydrophobicity_Score': solubility_features[1] if len(solubility_features) > 1 else None,  
            'Charge_Score': solubility_features[2] if len(solubility_features) > 2 else None  
        }  
        
        return properties  
        
    except Exception as e:  
        print(f"Error processing sequence: {sequence}")  
        print(f"Error message: {str(e)}")  
        return None  

def process_sequences(file_path):  
    results = []  
    sequences = list(SeqIO.parse(file_path, 'fasta'))  
    
    print(f"Processing {len(sequences)} sequences from {file_path}...")  
    
    for i, seq in enumerate(sequences, 1):  
        if i % 100 == 0:  
            print(f"Processed {i}/{len(sequences)} sequences...")  
            
        properties = calculate_sequence_properties(seq.seq)  
        if properties:  
            properties['Sequence_ID'] = seq.id  
            properties['Sequence'] = str(seq.seq)  

            helix, turn, sheet = properties.pop('Secondary_Structure')  
            properties['Helix_Fraction'] = helix  
            properties['Turn_Fraction'] = turn  
            properties['Sheet_Fraction'] = sheet  

            reduced, oxidized = properties.pop('Molar_Extinction')  
            properties['Extinction_Reduced'] = reduced  
            properties['Extinction_Oxidized'] = oxidized  
            
            results.append(properties)  
    
    return results  

def main():  
    try:  

        tox_pred_file = 'TasToxpred_test_predictions.csv'  
        if not os.path.exists(tox_pred_file):  
            print(f"Error: {tox_pred_file} not found")  
            return  
        tox_predictions = pd.read_csv(tox_pred_file)  

        results = process_sequences('TasToxPred/dataset/all_test.fasta')  

        if results:  

            physchem_df = pd.DataFrame(results)  

            columns_order = [  
                'Sequence_ID', 'Sequence',  
                'GRAVY', 'Isoelectric_Point', 'Charge_at_pH7', 'Molecular_Weight',  
                'Aromaticity', 'Instability_Index', 'Aliphatic_Index',  
                'Charge_Density', 'Hydrophobic_Ratio', 'Hydrophobic_Moment',  
                'Solubility_Score', 'Hydrophobicity_Score', 'Charge_Score',  
                'Helix_Fraction', 'Turn_Fraction', 'Sheet_Fraction',  
                'Extinction_Reduced', 'Extinction_Oxidized'  
            ]  

            for col in columns_order:  
                if col not in physchem_df.columns:  
                    physchem_df[col] = np.nan  
                    
            physchem_df = physchem_df[columns_order]  

            physchem_features = physchem_df.drop(['Sequence_ID', 'Sequence'], axis=1)  

            if len(tox_predictions) != len(physchem_features):  
                print("Error: Number of sequences doesn't match between toxicity predictions and physicochemical properties")  
                return  

            final_results = pd.concat([tox_predictions, physchem_features], axis=1)  

            final_results.to_csv(tox_pred_file, index=False)  
            print(f"\nResults have been appended to: {tox_pred_file}")  

            print("\nBasic statistics for physicochemical properties:")  
            numerical_columns = columns_order[2:]  
            for col in numerical_columns:  
                print(f"\n{col}:")  
                stats = final_results[col].describe()  
                print(stats)  
        else:  
            print("No results were generated. Please check your input file and sequences.")  
            
    except Exception as e:  
        print(f"An error occurred during execution: {str(e)}")  

if __name__ == "__main__":  
    main()
