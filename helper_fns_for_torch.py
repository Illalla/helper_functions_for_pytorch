import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm
import itertools
from PIL import Image
import random

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
          X, y = X.to(device), y.to(device)
          test_pred = model(X)
          test_loss += loss_fn(test_pred, y)
          test_metric += metric(test_pred.argmax(dim=1), y)
        test_loss /= len(test_dataloader)
        test_metric /= len(test_dataloader)
      print(f'\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | {str(metric)}: {test_metric:.4f}')


def show_random_images(path, class_name='*', nrows=3, ncols=3, figsize=(12, 8)):
  """Shows random images given path and class name. The dataset must be should be in
  format 'path/*/classname/*.jpg'"""
  images_paths = list(Path(path).glob(f'*/{class_name}/*.jpg'))
  random_paths = random.sample(images_paths, k=nrows*ncols)
  
  fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
  for i, j in itertools.product(range(nrows), range(ncols)):
    img_path = random_paths[i+j]
    img = Image.open(img_path)
    ax[i, j].imshow(img)
    ax[i, j].title.set_text(f'Image size: {img.size}\nImage class: {img_path.parent.stem}')
    ax[i, j].axis(False)
