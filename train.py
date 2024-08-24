import gc
import os
import sys
import shutil
import time
sys.path.append(os.path.abspath('d:/Psychotherapy-app/model'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from model.classifier import KoBERTforSequenceClassfication
from model.dataloader import WellnessTextClassificationDataset


def save_checkpoint(save_path, model, optimizer, epoch, loss, retries=5):
    for attempt in range(retries):
        try:
            temp_save_path = save_path + '.tmp'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, temp_save_path)
            shutil.move(temp_save_path, save_path)
            return
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            time.sleep(1)
    print("Failed to save checkpoint after multiple retries")

def train(device, epoch, model, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for data in train_loader:
            # Ensure data is on the correct device
            inputs = {key: val.to(device) for key, val in data.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[0]
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

    return np.mean(losses)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    root_path = "."
    data_path = f"{root_path}/data/wellness_dialog_for_text_classification_all.txt"
    checkpoint_path = f"{root_path}/checkpoint"
    
    os.makedirs(checkpoint_path, exist_ok=True)
    save_ckpt_path = f"{checkpoint_path}/kobert-wellness-text-classification.pth"

    n_epoch = 100
    batch_size = 4
    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)
    learning_rate = 5e-6

    dataset = WellnessTextClassificationDataset(file_path=data_path, device=device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KoBERTforSequenceClassfication()
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    pre_epoch, pre_loss = 0, 0
    if os.path.isfile(save_ckpt_path):
        checkpoint = torch.load(save_ckpt_path, map_location=device)
        pre_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}")

    losses = []
    offset = pre_epoch
    for step in range(n_epoch):
        epoch = step + offset
        loss = train(device, epoch, model, optimizer, train_loader)
        losses.append(loss)

        # Save checkpoint at the end of each epoch
        save_checkpoint(save_ckpt_path, model, optimizer, epoch, loss)
