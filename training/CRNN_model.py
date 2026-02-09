import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os

class CRNN(nn.Module):
    def __init__(self, img_height=32, num_channels=1, num_classes=37, hidden_size=256):
        super(CRNN, self).__init__()
        assert img_height % 32 == 0
        self.cnn = nn.Sequential(
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
        self.lstm1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=False)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=False)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        conv = conv.view(batch, channels * height, width)  # (B, C*H, W)
        conv = conv.permute(2, 0, 1)
        lstm_out, _ = self.lstm1(conv)
        lstm_out, _ = self.lstm2(lstm_out)
        output = self.fc(lstm_out)  # (W, B, num_classes)
        
        return output
    
def ctc_greedy_decoder(outputs, alphabet):
    _, preds = torch.max(outputs, dim=2)
    preds = preds.transpose(1, 0).cpu().numpy()
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