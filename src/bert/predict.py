import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config

# Prediction function
def predict(model, dataloader):
    model.eval()
    preds_list = []
    for batch in dataloader:
        batch_input_ids, batch_attention_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_list.extend(preds)
    return preds_list

# Confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    sns.heatmap(cm, annot=True, cmap="Purples", fmt='g', cbar=False, annot_kws={"size": 30})
    ax.xaxis.set_ticklabels(labels, fontsize=16)
    ax.yaxis.set_ticklabels(labels, fontsize=14.5)
    ax.set_ylabel('True', fontsize=25)
    ax.set_xlabel('Predicted', fontsize=25)
    plt.show()
