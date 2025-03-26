import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, RLock
from glob import glob
import tempfile
import argparse
import signal
import sys

from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize2, tokenize3

def cleanup_temp_files():
    """Clean up any leftover temporary files from previous runs."""
    patterns = ['./data/temp_*.txt', './data/done_*.marker', './data/output.txt', './data/output_seq.txt']
    for pattern in patterns:
        for file in glob(pattern):
            try:
                os.remove(file)
                print(f"Cleaned up file: {file}")
            except:
                pass

def process_file(file_tuple, file_idx):
    """Process a single file tuple and return the results."""
    # Create a unique temporary output file
    output_file = f'./data/temp_{file_idx:04d}.txt'
    # Process just this one file tuple
    results = tokenize3([file_tuple], output=output_file, skip_Nones=True)
    return results, output_file, file_idx

def signal_handler(sig, frame):
    """Handle interruption signals gracefully."""
    print('\nProgram interrupted! Run again to restart from the beginning.')
    sys.exit(0)

def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Add command line argument for parallel vs sequential processing
    parser = argparse.ArgumentParser(description='Tokenize ASAP dataset')
    parser.add_argument('--sequential', action='store_true', 
                        help='Use sequential processing instead of parallel')
    args = parser.parse_args()
    
    # Always clean up before starting
    print("Cleaning up any previous files...")
    cleanup_temp_files()
    
    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    BASE = "./asap-dataset-master/"
    df = pd.read_csv('asap-dataset-master/metadata.csv')

    datafiles = []

    for i, row in df.iterrows():
        file1 = BASE + row['midi_performance']
        file2 = BASE + row['midi_score']
        file3 = BASE + row['performance_annotations']
        file4 = BASE + row['midi_score_annotations']

        datafiles.append((file1, file2, file3, file4))

    print('Tokenizing data in parallel; will be written to output.txt')
    
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    
    if args.sequential:
        print('Using sequential processing to avoid potential misalignments')
        # Process files sequentially
        results = []
        
        final_output = './data/output_seq.txt'
        
        for i, file_tuple in enumerate(tqdm(datafiles, desc="Processing files")):
            # Process directly to final output file
            result = tokenize3([file_tuple], output=final_output, skip_Nones=(i > 0))
            results.append(result)
    else:
        print('Using parallel processing')
        # Determine number of processes to use (limit to avoid memory issues)
        num_processes = min(os.cpu_count() or 4, 8)  # Cap at 8 processes to avoid memory issues
        print(f'Using {num_processes} processes for parallelization')
        
        # Process files individually in parallel
        results = []
        temp_files = []
        
        # Create a pool with a lock for tqdm
        with Pool(processes=num_processes) as pool:
            # Create arguments for each file
            file_args = [(datafiles[i], i) for i in range(len(datafiles))]
            
            # Process each file in parallel with progress tracking
            for result, temp_file, file_idx in tqdm(
                pool.starmap(process_file, file_args), 
                total=len(datafiles),
                desc="Processing files"
            ):
                results.append((result, file_idx))
                temp_files.append((temp_file, file_idx))
        
        # Sort results and temp_files by original file index to maintain order
        results.sort(key=lambda x: x[1])
        temp_files.sort(key=lambda x: x[1])
        
        # Extract just the results
        results = [r[0] for r in results]
        temp_files = [t[0] for t in temp_files]
        
        # Combine output files in the correct order
        with open('./data/output.txt', 'w') as outfile:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as infile:
                        outfile.write(infile.read())
                    # Clean up temporary file
                    os.remove(temp_file)
    
    # Aggregate results from all processes
    seq_count = sum(r[0] for r in results)
    rest_count = sum(r[1] for r in results)
    too_short = sum(r[2] for r in results)
    too_long = sum(r[3] for r in results)
    too_manyinstr = sum(r[4] for r in results)
    discarded_seqs = sum(r[5] for r in results)
    truncations = sum(r[6] for r in results)
    
    rest_ratio = round(100*float(rest_count)/(seq_count*M),2)

    trunc_type = 'duration' #'interarrival' if args.interarrival else 'duration'
    trunc_ratio = round(100*float(truncations)/(seq_count*M),2)

    print('Tokenization complete.')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {too_short+too_long} event sequences')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {too_manyinstr} too many instruments')
    print(f'  => Discarded {discarded_seqs} training sequences')
    print(f'  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)')

    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    main()
