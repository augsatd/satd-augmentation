import torch
import time
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import config

# Initialize the model, optimizer, and scheduler
def initialize_model(train_dataloader, epochs=4):
    model = BertClassifier(freeze_bert=False)
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return model, optimizer, scheduler

# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=4):
    loss_fn = nn.CrossEntropyLoss()
    for epoch_i in range(epochs):
        print(f'Epoch {epoch_i + 1}/{epochs}')
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(config.device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss:.2f}')

        # Validation
        model.eval()
        val_loss = []
        val_accuracy = []
        for batch in val_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(config.device) for t in batch)
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        print(f'Validation loss: {val_loss:.2f}, Validation accuracy: {val_accuracy:.2f}%')
    print("Training complete!")
