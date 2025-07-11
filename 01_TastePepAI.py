import os  
import sys  
import glob  
import shutil  
import subprocess  
import time  
from datetime import datetime  

def print_welcome_message():  

    print("=" * 100)  
    print("Welcome to LA-VAE)")  
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

def get_user_inputs():  

    use_multiple = input("\nWould you like to enter multiple taste patterns? (yes/no): ").lower()  
    
    if use_multiple == 'yes':  
        while True:  
            patterns = input("Enter your desired taste patterns separated by commas (e.g., x1xxx,xxx1x,xxxx1): ")  
            pattern_list = [pattern.strip() for pattern in patterns.split(',')]  

            valid_patterns = True  
            for pattern in pattern_list:  
                if len(pattern) != 5 or not all(char in '01x' for char in pattern):  
                    print(f"Invalid pattern: {pattern}. Each pattern must be exactly five characters (0, 1, or x).")  
                    valid_patterns = False  
                    break  
            
            if valid_patterns:  
                break  
    else:  
        while True:  
            patterns = input("Enter your desired taste profile (e.g., xxxx1): ")  
            if len(patterns) != 5 or not all(char in '01x' for char in patterns):  
                print("Invalid input. Please ensure you enter exactly five characters, each either 0, 1, or x.")  
            else:  
                break  
    
    return f"{use_multiple}\n{patterns}\n"  

def run_command(command, description, inputs=None):  

    print(f"\n{'-'*80}")  
    print(f"Starting {description} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
    print(f"Command: {' '.join(command)}")  
    print(f"{'-'*80}\n")  
    
    try:  
        if inputs:  
            process = subprocess.Popen(  
                command,  
                stdin=subprocess.PIPE,  
                stdout=subprocess.PIPE,  
                stderr=subprocess.PIPE,  
                universal_newlines=True  
            )  
            
            stdout, stderr = process.communicate(inputs)  
            print(stdout)  
            
            if process.returncode != 0:  
                print(f"Error occurred: {stderr}")  
                return False  
        else:  
            process = subprocess.Popen(  
                command,  
                stdout=subprocess.PIPE,  
                stderr=subprocess.PIPE,  
                universal_newlines=True  
            )  
            
            while True:  
                output = process.stdout.readline()  
                if output == '' and process.poll() is not None:  
                    break  
                if output:  
                    print(output.strip())  
            
            return_code = process.poll()  
            if return_code != 0:  
                _, stderr = process.communicate()  
                print(f"Error occurred: {stderr}")  
                return False  
                
        return True  
        
    except Exception as e:  
        print(f"Error executing {description}: {str(e)}")  
        return False  

def find_latest_fasta():  

    result_dirs = glob.glob('results_*')  
    if not result_dirs:  
        return None  
    
    latest_dir = max(result_dirs)  
    fasta_files = glob.glob(os.path.join(latest_dir, '*.fasta'))  
    
    if not fasta_files:  
        return None  
        
    return fasta_files[0]  

def main():  
    print("\n=== Starting TastePep-Toxicity Pipeline ===")  
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
    
    print_welcome_message()  
    
    user_inputs = get_user_inputs()  

    success = run_command(  
        ['python', '02_LA-VAE.py'],  
        'Taste Peptide Generation and Clustering',  
        inputs=user_inputs  
    )  
    
    if not success:  
        print("Error: Taste peptide generation failed")  
        sys.exit(1)  

    fasta_file = find_latest_fasta()  
    if not fasta_file:  
        print("Error: No fasta file generated")  
        sys.exit(1)  

    try:  
        os.makedirs('TasToxPred/dataset', exist_ok=True)  
        shutil.copy(fasta_file, 'TasToxPred/dataset/all_test.fasta')  
        print(f"\nCopied {fasta_file} to TasToxPred/dataset/all_test.fasta")  
    except Exception as e:  
        print(f"Error copying fasta file: {str(e)}")  
        sys.exit(1)  

    success = run_command(  
        ['python', '03-TasToxPred.py'],  
        'Toxicity Prediction'  
    )  
    
    if not success:  
        print("Error: Toxicity prediction failed")  
        sys.exit(1)  

    success = run_command(  
        ['python', '04_compute_physicochemical.py'],  
        'Physicochemical Properties Calculation'  
    )  

    if not success:  
        print("Error: Physicochemical properties calculation failed")  
        sys.exit(1)  
    
    latest_dir = max(glob.glob('results_*'))  
    try:  
        if os.path.exists('TasToxpred_test_predictions.csv'):  
            shutil.move(  
                'TasToxpred_test_predictions.csv',  
                os.path.join(latest_dir, 'TasToxpred_test_predictions.csv')  
            )  
            print(f"\nMoved toxicity prediction results to {latest_dir}")  

        if os.path.exists('sequence_properties.csv'):  
            shutil.move(  
                'sequence_properties.csv',  
                os.path.join(latest_dir, 'sequence_properties.csv')  
            )  
            print(f"Moved physicochemical properties results to {latest_dir}")  
    except Exception as e:  
        print(f"Error moving results: {str(e)}")  
       
    print("\n=== Pipeline Completed Successfully ===")  
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
    print(f"Results directory: {latest_dir}")  

    try:  
        if os.path.exists('catboost_info'):  
            shutil.rmtree('catboost_info')  
            print("Cleaned up: removed catboost_info directory")  
    except Exception as e:  
        print(f"Warning: Could not remove catboost_info directory: {str(e)}")  

if __name__ == '__main__':  
    main()
