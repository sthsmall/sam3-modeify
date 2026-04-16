#!/usr/bin/env python3
"""
Evaluate SAM3 tracking results using TrackEval.
"""

import os
import sys
import shutil
import configparser
from datetime import datetime
from trackeval import Evaluator
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA, CLEAR, Identity

# Add the current directory to path to import get_prompt_from_sequence
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_sam3_viso import get_prompt_from_sequence

# Configuration
DATASET_ROOT = r"d:\projects\specific\torch-learning\sam3\dataset\laboratory_data\laboratory_data"
RESULTS_DIR = r"d:\projects\specific\torch-learning\sam3\eval\results"
EVAL_OUTPUT_DIR = r"d:\projects\specific\torch-learning\sam3\eval\output"

# Ensure output directory exists
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

def get_seq_length(gt_folder, seq_name):
    """Get sequence length from seqinfo.ini"""
    ini_file = os.path.join(gt_folder, seq_name, 'seqinfo.ini')
    if os.path.exists(ini_file):
        ini_data = configparser.ConfigParser()
        ini_data.read(ini_file)
        return int(ini_data['Sequence']['seqLength'])
    return None

def prepare_tracker_results(tracker_name, seq_name):
    """Prepare tracker results in the correct format for TrackEval"""
    # Check if the result file already exists in the tracker directory
    tracker_dir = os.path.join(RESULTS_DIR, tracker_name)
    result_file = os.path.join(tracker_dir, f"{seq_name}.txt")
    
    if os.path.exists(result_file):
        return True
    return False

def evaluate_sequence(gt_folder, tracker_folder, seq_name, tracker_name):
    """Evaluate a single sequence and return metrics"""
    # Get sequence length
    seq_length = get_seq_length(gt_folder, seq_name)
    
    # Create output directory for this sequence
    sequence_output_dir = os.path.join(EVAL_OUTPUT_DIR, tracker_name)
    os.makedirs(sequence_output_dir, exist_ok=True)
    
    # Configuration for TrackEval
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 4,
        'BREAK_ON_ERROR': False,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
        'OUTPUT_FOLDER': sequence_output_dir,
    }
    
    # Dataset configuration
    dataset_config = {
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': tracker_folder,
        'TRACKER_SUB_FOLDER': '',
        'TRACKERS_TO_EVAL': [tracker_name],
        'SPLIT_TO_EVAL': 'train',
        'CLASSES_TO_EVAL': ['pedestrian'],
        'INPUT_AS_ZERO_BASED': False,
        'SEQ_INFO': {seq_name: seq_length},
        'SKIP_SPLIT_FOL': True,
    }
    
    # Metric configuration
    metric_config = {
        'THRESHOLD': 0.5,
    }
    
    # Create dataset and metrics
    dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = [HOTA(metric_config), CLEAR(metric_config), Identity(metric_config)]
    
    # Run evaluation
    evaluator = Evaluator(eval_config)
    metrics_results = evaluator.evaluate(dataset_list, metrics_list)
    
    # Extract metrics for this sequence
    if metrics_results and len(metrics_results) > 0:
        # Get dataset results
        dataset_results = metrics_results[0]
        if 'MotChallenge2DBox' in dataset_results:
            mot_results = dataset_results['MotChallenge2DBox']
            if isinstance(mot_results, dict):
                if tracker_name in mot_results:
                    tracker_data = mot_results[tracker_name]
                    if isinstance(tracker_data, dict):
                        # Check if the sequence name is in the tracker data
                        if seq_name in tracker_data:
                            seq_data = tracker_data[seq_name]
                            if isinstance(seq_data, dict):
                                if 'pedestrian' in seq_data:
                                    pedestrian_data = seq_data['pedestrian']
                                    if isinstance(pedestrian_data, dict):
                                        # Create a combined metrics dictionary
                                        combined_metrics = {}
                                        
                                        # Add CLEAR metrics
                                        if 'CLEAR' in pedestrian_data:
                                            clear_data = pedestrian_data['CLEAR']
                                            if isinstance(clear_data, dict):
                                                combined_metrics.update(clear_data)
                                        
                                        # Add Identity metrics
                                        if 'Identity' in pedestrian_data:
                                            identity_data = pedestrian_data['Identity']
                                            if isinstance(identity_data, dict):
                                                combined_metrics.update(identity_data)
                                        
                                        # Add HOTA metrics
                                        if 'HOTA' in pedestrian_data:
                                            hota_data = pedestrian_data['HOTA']
                                            if isinstance(hota_data, dict):
                                                combined_metrics.update(hota_data)
                                        
                                        # Save sequence metrics to file
                                        sequence_metrics_file = os.path.join(sequence_output_dir, f'{seq_name}_metrics.txt')
                                        with open(sequence_metrics_file, 'w') as f:
                                            f.write(f"Metrics for {seq_name} (Type: {tracker_name})\n")
                                            f.write("MOTA: {:.4f}\n".format(combined_metrics.get('MOTA', 0)))
                                            f.write("MOTP: {:.4f}\n".format(combined_metrics.get('MOTP', 0)))
                                            f.write("IDF1: {:.4f}\n".format(combined_metrics.get('IDF1', 0)))
                                            f.write("IDP: {:.4f}\n".format(combined_metrics.get('IDP', 0)))
                                            f.write("IDR: {:.4f}\n".format(combined_metrics.get('IDR', 0)))
                                            f.write("IDs: {:.4f}\n".format(combined_metrics.get('IDs', 0)))
                                            f.write("FM: {:.4f}\n".format(combined_metrics.get('Frag', 0)))
                                            f.write("MT: {:.4f}\n".format(combined_metrics.get('MT', 0)))
                                            f.write("ML: {:.4f}\n".format(combined_metrics.get('ML', 0)))
                                            f.write("FP: {:.4f}\n".format(combined_metrics.get('CLR_FP', 0)))
                                            f.write("FN: {:.4f}\n".format(combined_metrics.get('CLR_FN', 0)))
                                        
                                        print(f"Sequence metrics saved to {sequence_metrics_file}")
                                        return combined_metrics
    
    return None

def calculate_overall_metrics(metrics_list):
    """Calculate overall metrics from a list of sequence metrics"""
    if not metrics_list:
        return None
    
    # Initialize overall metrics
    overall = {}
    
    # Get all metric keys from the first result
    if metrics_list[0]:
        metric_keys = metrics_list[0].keys()
        
        for key in metric_keys:
            # Check if this metric is a number
            if isinstance(metrics_list[0][key], (int, float)):
                # Calculate average
                values = [m[key] for m in metrics_list if m and key in m]
                if values:
                    overall[key] = sum(values) / len(values)
    
    return overall

def main():
    """Main function"""
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR) or len(os.listdir(RESULTS_DIR)) == 0:
        print(f"No results found in {RESULTS_DIR}. Please run run_sam3_viso.py first.")
        sys.exit(1)
    
    # Define sequence types to evaluate
    sequence_types = [
        'AIR-aircraft',
        'SAT-ship',
        'AIR-ship',
        'SAT-airplane'
    ]
    
    for seq_type in sequence_types:
        gt_folder = os.path.join(DATASET_ROOT, seq_type, 'train')
        
        # Check if GT folder exists
        if not os.path.exists(gt_folder):
            print(f"GT folder not found: {gt_folder}")
            continue
        
        # Get all sequence directories
        sequences = [d for d in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, d))]
        
        if not sequences:
            print(f"No sequences found for {seq_type}")
            continue
        
        # Store metrics for each sequence
        sequence_metrics = []
        
        # Evaluate the first 5 sequences for each type
        for i, seq in enumerate(sequences[:]):
            tracker_name = seq_type
            seq_name = seq
            
            # Prepare tracker results
            if not prepare_tracker_results(tracker_name, seq_name):
                print(f"Result file not found for {tracker_name}_{seq_name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating {tracker_name} on {seq_name} ({i+1}/5)")
            print(f"{'='*60}")
            
            # Evaluate
            try:
                metrics = evaluate_sequence(gt_folder, RESULTS_DIR, seq_name, tracker_name)
                print(f"DEBUG: metrics for {seq_name}: {metrics}")
                if metrics:
                    sequence_metrics.append(metrics)
                    print(f"DEBUG: sequence_metrics length: {len(sequence_metrics)}")
            except Exception as e:
                print(f"Error evaluating {tracker_name}_{seq_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate overall metrics for this type
        print(f"DEBUG: Final sequence_metrics length for {seq_type}: {len(sequence_metrics)}")
        if sequence_metrics:
            overall_metrics = calculate_overall_metrics(sequence_metrics)
            print(f"DEBUG: overall_metrics: {overall_metrics}")
            if overall_metrics:
                print(f"\n{'='*60}")
                print(f"Overall metrics for {seq_type}")
                print(f"{'='*60}")
                
                # Print overall metrics
                print("MOTA: {:.4f}".format(overall_metrics.get('MOTA', 0)))
                print("MOTP: {:.4f}".format(overall_metrics.get('MOTP', 0)))
                print("IDF1: {:.4f}".format(overall_metrics.get('IDF1', 0)))
                print("IDP: {:.4f}".format(overall_metrics.get('IDP', 0)))
                print("IDR: {:.4f}".format(overall_metrics.get('IDR', 0)))
                print("IDs: {:.4f}".format(overall_metrics.get('IDs', 0)))
                print("FM: {:.4f}".format(overall_metrics.get('Frag', 0)))
                print("MT: {:.4f}".format(overall_metrics.get('MT', 0)))
                print("ML: {:.4f}".format(overall_metrics.get('ML', 0)))
                print("FP: {:.4f}".format(overall_metrics.get('CLR_FP', 0)))
                print("FN: {:.4f}".format(overall_metrics.get('CLR_FN', 0)))
                
                # Save overall metrics to file
                overall_output_dir = os.path.join(EVAL_OUTPUT_DIR, seq_type)
                os.makedirs(overall_output_dir, exist_ok=True)
                
                # Get prompt for this sequence type
                prompt = get_prompt_from_sequence(seq_type)
                
                # Get current date and time
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                overall_file = os.path.join(overall_output_dir, 'overall_metrics.txt')
                with open(overall_file, 'w') as f:
                    f.write(f"Overall metrics for {seq_type}\n")
                    f.write(f"Date: {current_date}\n")
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"MOTA: {overall_metrics.get('MOTA', 0):.4f}\n")
                    f.write(f"MOTP: {overall_metrics.get('MOTP', 0):.4f}\n")
                    f.write(f"IDF1: {overall_metrics.get('IDF1', 0):.4f}\n")
                    f.write(f"IDP: {overall_metrics.get('IDP', 0):.4f}\n")
                    f.write(f"IDR: {overall_metrics.get('IDR', 0):.4f}\n")
                    f.write(f"IDs: {overall_metrics.get('IDs', 0):.4f}\n")
                    f.write(f"FM: {overall_metrics.get('Frag', 0):.4f}\n")
                    f.write(f"MT: {overall_metrics.get('MT', 0):.4f}\n")
                    f.write(f"ML: {overall_metrics.get('ML', 0):.4f}\n")
                    f.write(f"FP: {overall_metrics.get('CLR_FP', 0):.4f}\n")
                    f.write(f"FN: {overall_metrics.get('CLR_FN', 0):.4f}\n")
                
                print(f"\nOverall metrics saved to {overall_file}")
        else:
            print(f"DEBUG: No metrics collected for {seq_type}")

if __name__ == "__main__":
    main()
