import os
import json
import pickle
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_FEATURES_DIR = Path('/content/drive/MyDrive/Kelompok9/features')

if not os.path.exists(DEFAULT_FEATURES_DIR):
        DEFAULT_FEATURES_DIR = Path('/content/drive/My Drive/Kelompok9/features')

MAX_CONTENT_LEN = 28
MAX_SEQ_LEN = 30
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_features(path):
    return torch.load(path)

def encode_caption(caption_str, word2idx):
    tokens = caption_str.split()[:MAX_CONTENT_LEN]
    token_ids = [word2idx.get(w, UNK_IDX) for w in tokens]
    seq = [START_IDX] + token_ids + [END_IDX]
    pad_len = MAX_SEQ_LEN - len(seq)
    seq = seq + [PAD_IDX] * pad_len
    return seq

class CaptionDataset(Dataset):
    def __init__(self, image_names, captions_dict, features_dict, word2idx):
        self.features_dict = features_dict
        self.word2idx = word2idx
        self.samples = []
        for name in image_names:
            for caption in captions_dict[name]:
                self.samples.append((name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]
        feature = self.features_dict[image_name].float()
        caption_ids = torch.tensor(encode_caption(caption, self.word2idx), dtype=torch.int64)
        return feature, caption_ids

def extract_image_references(captions_dict, image_names):
    return [[c.split() for c in captions_dict[name]] for name in image_names]

def extract_feature_stack(features_dict, image_names):
    return torch.stack([features_dict[name].float() for name in image_names])

def get_loaders(batch_size, features_dir=DEFAULT_FEATURES_DIR):
    features_dir = Path(features_dir)
    splits = load_json(features_dir / 'splits.json')
    captions = load_json(features_dir / 'captions.json')
    features = load_features(features_dir / 'features.pt')
    vocab = load_pickle(features_dir / 'vocab.pkl')
    word2idx = vocab['word2idx']
    
    train_ds = CaptionDataset(splits['train'], captions, features, word2idx)
    val_ds = CaptionDataset(splits['val'], captions, features, word2idx)
    test_ds = CaptionDataset(splits['test'], captions, features, word2idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, vocab

def get_eval_set(split, features_dir=DEFAULT_FEATURES_DIR):
    features_dir = Path(features_dir)
    splits = load_json(features_dir / 'splits.json')
    captions = load_json(features_dir / 'captions.json')
    features = load_features(features_dir / 'features.pt')
    image_names = splits[split]
    
    feature_tensor = extract_feature_stack(features, image_names)
    references = extract_image_references(captions, image_names)
    
    return image_names, feature_tensor, references