import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
import numpy as np
import random
from datasets import load_from_disk

class OCRPreprocessing:
    def __init__(self, img_h=32, img_w=100, keep_aspect_ratio=False, normalize_mean=0.5, normalize_std=0.5):
        self.img_h = img_h
        self.img_w = img_w
        self.keep_aspect_ratio = keep_aspect_ratio
        if keep_aspect_ratio:
            self.transform=transforms.Compose([
                transforms.Grayscale(),
                ResizeKeepAspectRatio(img_h, img_w),
                transforms.ToTensor(),
                transforms.Normalize([normalize_mean], [normalize_std])
            ])
        else:
            self.transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize([normalize_mean], [normalize_std])
            ])
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image)
    
class ResizeKeepAspectRatio:
    def __init__(self, target_height, target_width, fill=0):
        self.target_height = target_height
        self.target_width = target_width
        self.fill = fill
    def __call__(self, img):
        # calculate new width to maintain aspect ratio
        w,h = img.size
        aspect_ratio = w/h
        n_width = max(1, int(self.target_height * aspect_ratio))
        #Resize
        img = img.resize((n_width, self.target_height), Image.LANCZOS)
        if n_width <= self.target_width:
            # pad to target width
            new_img = Image.new('L', (self.target_width, self.target_height), color=self.fill)
            new_img.paste(img, (0,0))
            img = new_img
        elif n_width > self.target_width:
            img = img.resize((self.target_width, self.target_height))
        return img
    
class AugmentedOCRPreprocessing(OCRPreprocessing):
    def __init__(self, *args, augment_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        #Augmentation transforms
        self.augment_prob = augment_prob
        self.augmentation = transforms.Compose([
            transforms.Grayscale(),
            ResizeKeepAspectRatio(self.img_h, self.img_w) if self.keep_aspect_ratio 
            else transforms.Resize((self.img_h, self.img_w)),
            RandomOCRAugmentation(augment_prob),  # Custom augmentations
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.augmentation(image)

class RandomOCRAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, img):
        # Random rotation (small angles only)
        if random.random() < self.prob:
            angle = random.uniform(-3, 3)  # ±3 degrees only
            img = img.rotate(angle, fillcolor=255)
        if random.random() < self.prob:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        if random.random() < self.prob:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        if random.random() < self.prob * 0.5:  # Less frequent
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
        if random.random() < self.prob * 0.3:  # Even less frequent
            img_array = np.array(img)
            noise = np.random.normal(0, 3, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        return img
    
class LabelPreprocessor:
    def __init__(self, alphabet=None, case_sensitive=False):
        self.case_sensitive = case_sensitive
        if alphabet is None:
            self.alphabet = None
        else:
            self.alphabet = alphabet
            self._create_mapping()
    def build_alphabet(self, labels):
        chars = set()
        for label in labels:
            if not self.case_sensitive:
                label = label.lower()
            chars.update(label)
        self.alphabet = ''.join(sorted(chars))
        self._create_mapping()
        print(f"Print alphabet with {len(self.alphabet)} characters")
        print(self.alphabet)
        
    def _create_mapping(self):
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        self.char_to_idx['<blank>'] = 0
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.alphabet) + 1 
        
    def encode(self, text):
        if not self.case_sensitive:
            text = text.lower()
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                print(f"Warning: Unknown character '{char}' in text '{text}'")
        return encoded
    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx in self.idx_to_char and idx != 0:  # Skip blank
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)
class MJSynthOCRDataset(Dataset):
    """
    Complete dataset class with all preprocessing for MJSynth
    """
    
    def __init__(self, 
                 hf_dataset,
                 label_preprocessor,
                 img_height=32,
                 img_width=100,
                 keep_aspect_ratio=False,
                 augment=False,
                 augment_prob=0.5):
        """
        Args:
            hf_dataset: HuggingFace dataset
            label_preprocessor: LabelPreprocessor instance
            img_height: Target image height
            img_width: Target image width
            keep_aspect_ratio: Whether to maintain aspect ratio
            augment: Whether to apply augmentations
            augment_prob: Probability of augmentation
        """
        self.dataset = hf_dataset
        self.label_preprocessor = label_preprocessor
        
        # Choose preprocessor
        if augment:
            self.img_preprocessor = AugmentedOCRPreprocessing(
                img_height, img_width, keep_aspect_ratio, augment_prob=augment_prob
            )
        else:
            self.img_preprocessor = OCRPreprocessing(
                img_height, img_width, keep_aspect_ratio
            )
        
        # Determine label key
        sample = self.dataset[0]
        self.label_key = 'label' if 'label' in sample else 'text'
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Preprocess image
        image = sample['image']
        image = self.img_preprocessor(image)
        
        # Preprocess label
        text = sample[self.label_key]
        encoded_label = self.label_preprocessor.encode(text)
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)
        
        return image, encoded_label, text
def collate_fn(batch):
    """
    Custom collate function for batching
    Handles variable-length sequences required by CTC
    """
    images, labels, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Labels remain as list (variable lengths)
    return images, labels, texts
def create_train_val_test_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test
    
    Args:
        dataset: HuggingFace dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed
        
    Returns:
        train_ds, val_ds, test_ds
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplitting dataset:")
    print(f"  Total:  {total_size:,}")
    print(f"  Train:  {train_size:,} ({100*train_ratio:.0f}%)")
    print(f"  Val:    {val_size:,} ({100*val_ratio:.0f}%)")
    print(f"  Test:   {test_size:,} ({100*(1-train_ratio-val_ratio):.0f}%)")
    
    # Shuffle
    dataset = dataset.shuffle(seed=seed)
    
    # Split
    train_ds = dataset.select(range(train_size))
    val_ds = dataset.select(range(train_size, train_size + val_size))
    test_ds = dataset.select(range(train_size + val_size, total_size))
    
    return train_ds, val_ds, test_ds

def create_train_val_test_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test
    
    Args:
        dataset: HuggingFace dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed
        
    Returns:
        train_ds, val_ds, test_ds
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplitting dataset:")
    print(f"  Total:  {total_size:,}")
    print(f"  Train:  {train_size:,} ({100*train_ratio:.0f}%)")
    print(f"  Val:    {val_size:,} ({100*val_ratio:.0f}%)")
    print(f"  Test:   {test_size:,} ({100*(1-train_ratio-val_ratio):.0f}%)")
    
    # Shuffle
    dataset = dataset.shuffle(seed=seed)
    
    # Split
    train_ds = dataset.select(range(train_size))
    val_ds = dataset.select(range(train_size, train_size + val_size))
    test_ds = dataset.select(range(train_size + val_size, total_size))
    
    return train_ds, val_ds, test_ds
def prepare_mjsynth_for_crnn():
    """
    Complete preprocessing pipeline for MJSynth dataset
    """
    
    print("="*70)
    print("STEP-BY-STEP PREPROCESSING FOR CRNN OCR")
    print("="*70)
    
    # ===== STEP 1: Load Dataset =====
    print("\n[STEP 1] Loading MJSynth Dataset...")
    try:
        ds = load_from_disk("../data/mjsynth_300k")
        print(f"✓ Loaded {len(ds):,} samples from disk")
    except:
        from datasets import load_dataset
        SUBSET_SIZE = 200000
        ds = load_dataset("priyank-m/MJSynth_text_recognition", split="train")
        ds = ds.shuffle(seed=42).select(range(SUBSET_SIZE))
        print(f"✓ Downloaded {len(ds):,} samples")
    
    # ===== STEP 2: Split Dataset =====
    print("\n[STEP 2] Splitting into Train/Val/Test...")
    train_ds, val_ds, test_ds = create_train_val_test_split(
        ds, train_ratio=0.8, val_ratio=0.1, seed=42
    )
    
    # ===== STEP 3: Build Alphabet =====
    print("\n[STEP 3] Building Alphabet from Training Data...")
    label_key = 'label' if 'label' in train_ds[0] else 'text'
    all_labels = [sample[label_key] for sample in train_ds]
    
    label_processor = LabelPreprocessor(case_sensitive=False)
    label_processor.build_alphabet(all_labels)
    
    print(f"✓ Alphabet size: {len(label_processor.alphabet)}")
    print(f"✓ Number of classes (with blank): {label_processor.num_classes}")
    
    # ===== STEP 4: Create Datasets with Preprocessing =====
    print("\n[STEP 4] Creating PyTorch Datasets...")
    
    # Training set with augmentation
    train_dataset = MJSynthOCRDataset(
        hf_dataset=train_ds,
        label_preprocessor=label_processor,
        img_height=32,
        img_width=100,
        keep_aspect_ratio=True,
        augment=True,  # Enable augmentation for training
        augment_prob=0.5
    )
    
    # Validation set without augmentation
    val_dataset = MJSynthOCRDataset(
        hf_dataset=val_ds,
        label_preprocessor=label_processor,
        img_height=32,
        img_width=100,
        keep_aspect_ratio=True,
        augment=False  # No augmentation for validation
    )
    
    # Test set without augmentation
    test_dataset = MJSynthOCRDataset(
        hf_dataset=test_ds,
        label_preprocessor=label_processor,
        img_height=32,
        img_width=100,
        keep_aspect_ratio=True,
        augment=False  # No augmentation for testing
    )
    
    print(f"✓ Train dataset: {len(train_dataset):,} samples (with augmentation)")
    print(f"✓ Val dataset:   {len(val_dataset):,} samples (no augmentation)")
    print(f"✓ Test dataset:  {len(test_dataset):,} samples (no augmentation)")
    
    # ===== STEP 5: Create DataLoaders =====
    print("\n[STEP 5] Creating DataLoaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader:   {len(val_loader)} batches")
    print(f"✓ Test loader:  {len(test_loader)} batches")
    
    # ===== STEP 6: Verify Preprocessing =====
    print("\n[STEP 6] Verifying Preprocessing...")
    
    # Get a batch
    images, labels, texts = next(iter(train_loader))
    
    print(f"✓ Batch shape: {images.shape}")
    print(f"  - Batch size: {images.shape[0]}")
    print(f"  - Channels: {images.shape[1]} (grayscale)")
    print(f"  - Height: {images.shape[2]} pixels")
    print(f"  - Width: {images.shape[3]} pixels")
    print(f"  - Pixel range: [{images.min():.2f}, {images.max():.2f}] (normalized)")
    
    print(f"\n✓ Sample labels (first 5):")
    for i in range(min(5, len(texts))):
        print(f"  '{texts[i]}' → {labels[i].tolist()[:10]}...")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE! Ready for training.")
    print("="*70)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'label_processor': label_processor,
        'num_classes': label_processor.num_classes,
        'alphabet': label_processor.alphabet
    }
    
if __name__ == '__main__':
    # Run the complete preprocessing pipeline
    data = prepare_mjsynth_for_crnn()
    
    # Save for later use
    import pickle
    with open('preprocessing_data.pkl', 'wb') as f:
        pickle.dump({
            'label_processor': data['label_processor'],
            'alphabet': data['alphabet'],
            'num_classes': data['num_classes']
        }, f)
    
    print("\n✓ Preprocessing data saved to: preprocessing_data.pkl")