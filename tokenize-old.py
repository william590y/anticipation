import os
import numpy as np
import pandas as pd
import gc
import time
import signal
import psutil
from glob import glob
from tqdm import tqdm
import multiprocessing as mp

from anticipation.config import *
from anticipation.tokenize import tokenize2, tokenize3

def batch_process(datafiles, output_path, batch_size=1, skip_Nones=True, timeout_seconds=300, start_from_batch=0):
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
    
    # Track problematic files to skip in future runs
    problematic_files = []
    
    # Process in batches
    for batch_start in tqdm(range(start_from_batch * batch_size, len(datafiles), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(datafiles))
        current_batch = datafiles[batch_start:batch_end]
        
        # Report memory usage before processing batch
        process = psutil.Process(os.getpid())
        print(f"\n[Batch {batch_start//batch_size + 1}/{(len(datafiles) + batch_size - 1)//batch_size}] "
              f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        
        # Log files in this batch
        print(f"Processing files {batch_start}-{batch_end-1}:")
        for i, file_tuple in enumerate(current_batch):
            print(f"  {i+1}. {os.path.basename(file_tuple[0])}")
        
        # Use temporary output file for this batch
        temp_output = f"{output_path}.temp_{batch_start}"
        
        # Create a Queue to get the result
        result_queue = mp.Queue()
        
        # Define a function that will run tokenize3 and put the result in the queue
        def target_func():
            try:
                # Process each file separately first to identify problematic files
                for i, file_tuple in enumerate(current_batch):
                    try:
                        # Debug file processing
                        print(f"Pre-processing file {i+1}/{len(current_batch)}: {os.path.basename(file_tuple[0])}")
                        single_result = tokenize3(
                            [file_tuple], 
                            output=f"{temp_output}.debug_{i}",
                            skip_Nones=True
                        )
                        # Clean up debug file
                        if os.path.exists(f"{temp_output}.debug_{i}"):
                            os.remove(f"{temp_output}.debug_{i}")
                    except Exception as e:
                        print(f"⚠️ Error pre-processing file {os.path.basename(file_tuple[0])}: {e}")
                        # Add to problematic files but continue with batch processing
                        problematic_files.append((file_tuple[0], str(e)))
                
                # Now process the entire batch
                print(f"Processing full batch {batch_start//batch_size + 1}...")
                result = tokenize3(
                    current_batch, 
                    output=temp_output,
                    skip_Nones=(batch_start > 0 or skip_Nones)
                )
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)
        
        # Start the process
        process = mp.Process(target=target_func)
        process.start()
        
        print(f"Started process for batch {batch_start//batch_size + 1} (PID: {process.pid})")
        process_start_time = time.time()
        
        # Wait for the process to complete or timeout
        process.join(timeout_seconds)
        
        # Check if the process timed out
        if process.is_alive():
            elapsed = time.time() - process_start_time
            print(f"\n⚠️ Batch {batch_start//batch_size + 1} timed out after {elapsed:.2f} seconds. Terminating process...")
            
            # Try graceful termination first
            process.terminate()
            for _ in range(5):  # Wait up to 5 seconds for termination
                if not process.is_alive():
                    break
                time.sleep(1)
            
            # If process is still alive, force kill it
            if process.is_alive():
                print("Process didn't terminate gracefully. Forcing termination...")
                try:
                    if hasattr(process, 'kill'):
                        process.kill()
                    else:
                        os.kill(process.pid, signal.SIGKILL)
                except Exception as kill_error:
                    print(f"Error during force kill: {kill_error}")
            
            # Wait to ensure resources are released
            time.sleep(3)
            
            # Clean up temporary file if it exists
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                    print(f"Removed temporary file {temp_output}")
                except Exception as e:
                    print(f"Error removing temp file: {e}")
                    
            # Force garbage collection
            gc.collect()
            print(f"Memory after GC: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            continue
        
        # Check if an exception was raised
        if result_queue.empty():
            print(f"\n⚠️ Error processing batch {batch_start//batch_size + 1}: No result returned")
            # Clean up temporary file if it exists
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except Exception as e:
                    print(f"Error removing temp file: {e}")
            
            # Force garbage collection
            gc.collect()
            continue
        
        result_or_exception = result_queue.get()
        if isinstance(result_or_exception, Exception):
            print(f"\n⚠️ Error processing batch {batch_start//batch_size + 1}: {result_or_exception}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except Exception as e:
                    print(f"Error removing temp file: {e}")
            
            # Force garbage collection
            gc.collect()
            continue
        
        # Update totals
        seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations = result_or_exception
        total_seq_count += seq_count
        total_rest_count += rest_count
        total_too_short += too_short
        total_too_long += too_long
        total_too_manyinstr += too_manyinstr
        total_discarded_seqs += discarded_seqs
        total_truncations += truncations
        
        # Show stats for this batch
        print(f"Batch {batch_start//batch_size + 1} stats: {seq_count} sequences, {rest_count} rests")
        
        # Append this batch's output to the main output file
        try:
            with open(temp_output, 'r') as temp_file:
                with open(output_path, 'a' if os.path.exists(output_path) else 'w') as main_file:
                    for line in temp_file:
                        main_file.write(line)
            print(f"Successfully appended batch output to {output_path}")
        except Exception as e:
            print(f"⚠️ Error appending batch output: {e}")
        
        # Remove temporary file
        try:
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as e:
            print(f"Error removing temp file: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Save checkpoint of current batch
        with open(f"{output_path}.checkpoint", "w") as f:
            f.write(str((batch_start // batch_size) + 1))
        
        # Sleep a bit to ensure resources are freed
        time.sleep(1)
    
    # Report problematic files
    if problematic_files:
        print("\nProblematic files encountered:")
        for filename, error in problematic_files:
            print(f"  - {os.path.basename(filename)}: {error}")
        
        # Save list of problematic files
        with open(f"{output_path}.problematic_files", "w") as f:
            for filename, error in problematic_files:
                f.write(f"{filename},{error}\n")
    
    return (
        total_seq_count, 
        total_rest_count, 
        total_too_short, 
        total_too_long, 
        total_too_manyinstr, 
        total_discarded_seqs, 
        total_truncations
    )

def load_checkpoint(checkpoint_file):
    """Load the last processed batch from checkpoint"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return 0

def main():
    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    # Ensure output directory exists
    os.makedirs('./data', exist_ok=True)
    
    output_file = './data/output.txt'
    checkpoint_file = f"{output_file}.checkpoint"
    
    # Check for checkpoint
    start_from_batch = load_checkpoint(checkpoint_file)
    
    if start_from_batch > 0:
        print(f"Resuming from batch {start_from_batch}")
        # Don't remove existing output file when resuming
        if not os.path.exists(output_file):
            print("Warning: Checkpoint found but output file doesn't exist. Starting from the beginning.")
            start_from_batch = 0
    else:
        # Clear any existing output file
        if os.path.exists(output_file):
            os.remove(output_file)
            print("Removed existing output file.")

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

    # Display system info before starting
    process = psutil.Process(os.getpid())
    print(f"Initial memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"CPU count: {os.cpu_count()}")

    # Set smaller batch size for easier recovery
    batch_size = 2  # Smaller batch size
    result = batch_process(
        all_files, 
        output_file, 
        batch_size=batch_size, 
        timeout_seconds=240, 
        start_from_batch=start_from_batch
    )
    
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

    # Remove checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    main()