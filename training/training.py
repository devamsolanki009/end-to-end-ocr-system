import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from CRNN_model import ctc_greedy_decoder

def train_one_epoch(model, train_loader, criterion, optimizer, device, alphabet, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets, texts) in enumerate(progress_bar):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)  # (seq_len, batch, num_classes)
        
        # Log softmax for CTC
        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        
        # Prepare for CTC loss
        batch_size = images.size(0)
        input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat(targets)
        
        # CTC loss
        loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        # Calculate accuracy (on a subset for speed)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                preds = ctc_greedy_decoder(outputs, alphabet)
                for pred, true_text in zip(preds[:10], texts[:10]):
                    if pred.lower() == true_text.lower():
                        correct += 1
                    total += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy
def validate(model, val_loader, criterion, device, alphabet):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets, texts in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            # CTC loss
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            targets_concat = torch.cat(targets)
            
            loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            preds = ctc_greedy_decoder(outputs, alphabet)
            
            # Calculate accuracy
            for pred, true_text in zip(preds, texts):
                all_predictions.append(pred)
                all_ground_truths.append(true_text)
                
                if pred.lower() == true_text.lower():
                    correct += 1
                total += 1
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    # Show some examples
    print("\nSample predictions:")
    for i in range(min(10, len(all_predictions))):
        status = "✓" if all_predictions[i].lower() == all_ground_truths[i].lower() else "✗"
        print(f"  {status} True: '{all_ground_truths[i]}' | Pred: '{all_predictions[i]}'")
    
    return avg_loss, accuracy