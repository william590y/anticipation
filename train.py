import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm

from anticipation.config import *
from anticipation.vocab import *

class ScorePredictionDataset(Dataset):
    def __init__(self, filepath, window_size=1017):
        self.pairs = []
        self.window_size = window_size
        
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r') as f:
            sequences = [list(map(int, line.strip().split())) for line in tqdm(f.readlines())]
        
        print("Processing sequences and building training pairs...")
        for seq_idx, seq in enumerate(tqdm(sequences)):
            # First token is the control token (typically ANTICIPATE)
            z = [seq[0]]
            tokens = seq[1:]  # Skip the control token
            
            # Process each position where we need to predict a score token
            for pos in range(len(tokens)):
                target = tokens[pos]
                
                # We only want to predict score tokens (< CONTROL_OFFSET)
                if target >= CONTROL_OFFSET | target == SEPARATOR:
                    continue
                
                # Ensure we're at the start of a triplet for position context
                triplet_pos = pos % 3
                
                # Build context: limited history window matching add_token
                lookback = max(pos - self.window_size, 0)
                history = tokens[lookback:pos]
                
                # Relativize time tokens in history (every 3rd token starting at index 0)
                history_copy = history.copy()
                
                # Find time tokens in history (only actual time tokens, not offset ones)
                time_indices = [k for k in range(0, len(history_copy), 3) 
                if TIME_OFFSET <= history_copy[k] < TIME_OFFSET + MAX_TIME]
                
                if time_indices:
                    # Find minimum time value and subtract from all time tokens
                    time_tokens = [history_copy[k] for k in time_indices]
                    min_time = min(time_tokens) - TIME_OFFSET
                    
                    for k in time_indices:
                        history_copy[k] -= min_time
                
                # Get any previous tokens in the current triplet
                # This matches how add_token builds up triplets token by token
                new_tokens = []
                if triplet_pos > 0:
                    triplet_start = pos - triplet_pos
                    new_tokens = tokens[triplet_start:pos]
                
                # Add this sample
                self.pairs.append({
                    'z': z,
                    'history': history_copy,
                    'new_tokens': new_tokens,
                    'target': target,
                    'position': triplet_pos  # 0=time, 1=duration, 2=note
                })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # Combine z + history + new_tokens to match inference
        input_tokens = pair['z'] + pair['history'] + pair['new_tokens']
        
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'target': torch.tensor(pair['target'], dtype=torch.long),
            'position': torch.tensor(pair['position'], dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    positions = torch.stack([item['position'] for item in batch])
    
    # Pad inputs to same length
    max_len = max(len(ids) for ids in input_ids)
    input_ids_padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    for i, ids in enumerate(input_ids):
        input_ids_padded[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = True
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask,
        'targets': targets,
        'positions': positions
    }

def train(model, train_loader, val_loader, args):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get logits for the final position only (matching how inference works)
            last_token_logits = logits[torch.arange(logits.size(0)), attention_mask.sum(dim=1) - 1]
            
            loss = criterion(last_token_logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                last_token_logits = logits[torch.arange(logits.size(0)), attention_mask.sum(dim=1) - 1]
                loss = criterion(last_token_logits, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

def main():
    parser = argparse.ArgumentParser(description='Train an anticipatory music transformer')
    parser.add_argument('--train-file', type=str, default='./data/output.txt', 
                        help='Path to training data file')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--window-size', type=int, default=1017,
                        help='Context window size (default: 1017 to match inference)')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Creating dataset...")
    full_dataset = ScorePredictionDataset(args.train_file, window_size=args.window_size)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Dataset created with {train_size} training samples and {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model (replace with your actual model initialization)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")  # Replace with your model
    
    # Train model
    train(model, train_loader, val_loader, args)

if __name__ == '__main__':
    main()