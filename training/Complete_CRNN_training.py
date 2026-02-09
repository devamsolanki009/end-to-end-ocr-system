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
from CRNN_model import CRNN
from training import train_one_epoch, validate

def train_crnn_end_to_end(train_loader, val_loader, label_processor, num_epochs=50, learning_rate=0.001, device=None, save_dir='checkpoints'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*70)
    print("CRNN END-TO-END TRAINING PIPELINE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of classes: {label_processor.num_classes}")
    print(f"Alphabet: {label_processor.alphabet}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    print("\n[1] Initializing CRNN Model...")
    model = CRNN(
        img_height=32,
        num_channels=1,
        num_classes=label_processor.num_classes,
        hidden_size=256
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # ===== Loss and Optimizer =====
    print("\n[2] Setting up Loss and Optimizer...")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    print("✓ Loss: CTC Loss")
    print("✓ Optimizer: Adam")
    print("✓ LR Scheduler: ReduceLROnPlateau")
    
    # ===== Training Loop =====
    print("\n[3] Starting Training Loop...")
    
    best_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            label_processor.alphabet, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, label_processor.alphabet
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'alphabet': label_processor.alphabet,
            'num_classes': label_processor.num_classes
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(checkpoint, os.path.join(save_dir, 'best_checkpoint.pth'))
            print(f"  ✓ New best model! Accuracy: {val_acc:.2f}%")
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # ===== Training Complete =====
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoints saved in: {save_dir}/")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"✓ Training curves saved: {save_dir}/training_curves.png")
    
    return model, best_accuracy