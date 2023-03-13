import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm

def train_cls_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, metric, epochs, device):
    """Trains a model and evaluates it. Dataloaders must be batched."""
    n_batches = len(train_dataloader)
    model.to(device)
    for epoch in tqdm(range(epochs)):
      train_loss = 0
      model.train()
      for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % int(0.1*n_batches) == 0 :
          print(f'Batch: {batch}/{n_batches}')
      train_loss /= len(train_dataloader)

      test_loss, test_metric = 0, 0
      model.eval()
      with torch.inference_mode():
        for X, y in test_dataloader:
          test_pred = model(X)
          test_loss += loss_fn(test_pred, y)
          test_metric += metric(test_pred.argmax(dim=1), y)
        test_loss /= len(test_dataloader)
        test_metric /= len(test_dataloader)
      print(f'\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | {str(metric)}: {test_metric:.4f}')
