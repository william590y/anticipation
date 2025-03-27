import os
import numpy as np
import pandas as pd
import gc
from glob import glob
from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize2, tokenize3

def batch_process(datafiles, output_path, batch_size=4, skip_Nones=True):
    """Process files in small batches to avoid memory issues"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize counters
    total_seq_count = 0
    total_rest_count = 0
    total_too_short = 0
    total_too_long = 0
    total_too_manyinstr = 0
    total_discarded_seqs = 0
    total_truncations = 0
    
    # Process in batches
    for batch_start in tqdm(range(0, len(datafiles), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(datafiles))
        current_batch = datafiles[batch_start:batch_end]
        
        # Use temporary output file for this batch
        temp_output = f"{output_path}.temp_{batch_start}"
        
        # Process this batch
        result = tokenize3(
            current_batch, 
            output=temp_output,
            skip_Nones=(batch_start > 0 or skip_Nones)  # Only first batch needs headers
        )
        
        # Update totals
        seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations = result
        total_seq_count += seq_count
        total_rest_count += rest_count
        total_too_short += too_short
        total_too_long += too_long
        total_too_manyinstr += too_manyinstr
        total_discarded_seqs += discarded_seqs
        total_truncations += truncations
        
        # Append this batch's output to the main output file
        with open(temp_output, 'r') as temp_file:
            with open(output_path, 'a' if batch_start > 0 else 'w') as main_file:
                for line in temp_file:
                    main_file.write(line)
        
        # Remove temporary file
        os.remove(temp_output)
        
        # Force garbage collection
        gc.collect()
    
    return (
        total_seq_count, 
        total_rest_count, 
        total_too_short, 
        total_too_long, 
        total_too_manyinstr, 
        total_discarded_seqs, 
        total_truncations
    )

def main():
    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    # Ensure output directory exists
    os.makedirs('./data', exist_ok=True)
    
    # Clear any existing output file
    if os.path.exists('./data/output.txt'):
        os.remove('./data/output.txt')

    BASE = "./asap-dataset-master/"
    
    # Read metadata in chunks to avoid loading everything at once
    reader = pd.read_csv('asap-dataset-master/metadata.csv', chunksize=50)
    
    all_files = []
    for chunk in reader:
        for i, row in chunk.iterrows():
            file1 = BASE + row['midi_performance']
            file2 = BASE + row['midi_score']
            file3 = BASE + row['performance_annotations']
            file4 = BASE + row['midi_score_annotations']
            all_files.append((file1, file2, file3, file4))
    
    print('Tokenizing data; will be written to output.txt')
    print(f"Processing {len(all_files)} files in batches")

    result = batch_process(all_files, './data/output.txt', batch_size=5)
    
    seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations = result
    
    # Avoid division by zero
    if seq_count > 0:
        rest_ratio = round(100*float(rest_count)/(seq_count*M),2)
        trunc_ratio = round(100*float(truncations)/(seq_count*M),2)
    else:
        rest_ratio = 0
        trunc_ratio = 0

    trunc_type = 'duration'

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