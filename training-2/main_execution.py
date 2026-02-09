"""
Step 3: Complete End-to-End CRNN Training Pipeline
Integrates everything: preprocessing, model, training, evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==================== CRNN MODEL ====================

class CRNN(nn.Module):
    """
    Complete CRNN architecture for OCR
    """
    
    def __init__(self, img_height=32, num_channels=1, num_classes=37, hidden_size=256):
        super(CRNN, self).__init__()
        
        assert img_height % 16 == 0, "img_height must be divisible by 16"
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            # Layer 1: 64 filters
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2
            
            # Layer 2: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4
            
            # Layer 3: 256 filters
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 256 filters
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8
            
            # Layer 5: 512 filters
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 6: 512 filters
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16
            
            # Layer 7: 512 filters
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Recurrent layers
        self.lstm1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=False)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=False)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)  # (B, 512, H', W')
        
        # Map to sequence
        batch, channels, height, width = conv.size()
        conv = conv.view(batch, channels * height, width)  # (B, C*H, W)
        conv = conv.permute(2, 0, 1)  # (W, B, C*H)
        
        # RNN
        lstm_out, _ = self.lstm1(conv)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Output
        output = self.fc(lstm_out)  # (W, B, num_classes)
        
        return output


# ==================== CTC DECODER ====================

def ctc_greedy_decode(outputs, alphabet):
    """
    Greedy CTC decoding
    
    Args:
        outputs: Model outputs (seq_len, batch, num_classes)
        alphabet: String of characters
    
    Returns:
        List of decoded strings
    """
    # Get predictions
    _, preds = torch.max(outputs, dim=2)  # (seq_len, batch)
    preds = preds.transpose(1, 0).cpu().numpy()  # (batch, seq_len)
    
    decoded_texts = []
    
    for pred in preds:
        # Remove blanks and duplicates
        chars = []
        prev_idx = -1
        
        for idx in pred:
            if idx != 0 and idx != prev_idx:  # Not blank and not duplicate
                if idx - 1 < len(alphabet):
                    chars.append(alphabet[idx - 1])
            prev_idx = idx
        
        decoded_texts.append(''.join(chars))
    
    return decoded_texts


# ==================== TRAINING FUNCTIONS ====================

def train_one_epoch(model, train_loader, criterion, optimizer, device, alphabet, epoch):
    """
    Train for one epoch
    """
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
                preds = ctc_greedy_decode(outputs, alphabet)
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
            preds = ctc_greedy_decode(outputs, alphabet)
            
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


# ==================== COMPLETE TRAINING PIPELINE ====================

def train_crnn_end_to_end(train_loader, val_loader, label_processor, 
                         num_epochs=50, learning_rate=0.001, 
                         device=None, save_dir='checkpoints'):
    """
    Complete end-to-end training pipeline
    """
    
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
    
    # ===== Initialize Model =====
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


# ==================== INFERENCE FUNCTION ====================

def predict_text(model, image_path, label_processor, device, img_height=32, img_width=100):
    """
    Predict text from an image
    
    Args:
        model: Trained CRNN model
        image_path: Path to image file
        label_processor: LabelPreprocessor instance
        device: torch device
        img_height: Image height
        img_width: Image width
    
    Returns:
        Predicted text string
    """
    from PIL import Image
    import torchvision.transforms as transforms
    
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load and preprocess
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        prediction = ctc_greedy_decode(output, label_processor.alphabet)[0]
    
    return prediction


# ==================== MAIN EXECUTION ====================

def main():
    """
    Main execution function - runs the complete pipeline
    """
    
    # Import preprocessing
    from preprocessing.image_preprocessing import prepare_mjsynth_for_crnn
    
    print("\n" + "="*70)
    print("COMPLETE CRNN OCR PIPELINE")
    print("="*70)
    
    # Step 1: Prepare data
    print("\n>>> Preparing data with preprocessing...")
    data = prepare_mjsynth_for_crnn()
    
    # Step 2: Train model
    print("\n>>> Training CRNN model...")
    model, best_acc = train_crnn_end_to_end(
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        label_processor=data['label_processor'],
        num_epochs=30,  # Adjust as needed
        learning_rate=0.001,
        save_dir='checkpoints'
    )
    
    # Step 3: Test inference
    print("\n>>> Testing inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best checkpoint
    checkpoint = torch.load('checkpoints/best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Best model loaded (accuracy: {checkpoint['val_accuracy']:.2f}%)")
    print(f"✓ Ready for inference!")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nYou can now use predict_text() to recognize text in images:")
    print("  prediction = predict_text(model, 'image.jpg', label_processor, device)")


if __name__ == '__main__':
    main()