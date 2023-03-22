import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm
import itertools
from PIL import Image
import random

def train_cls_model(model,
                    train_dataloader,
                    test_dataloader,
                    optimizer,
                    loss_fn,
                    epochs,
                    device):
  model.to(device)
  loss_curves = {'train_loss': [],
                 'test_loss': [],
                 'train_acc': [],
                 'test_acc': []}
  
  for epoch in tqdm(range(epochs)):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in train_dataloader:
      X, y = X.to(device), y.to(device)
      y_pred_log = model(X)
      y_pred_class = torch.softmax(y_pred_log, dim=1).argmax(dim=1)
      loss = loss_fn(y_pred_log, y)
      train_loss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_acc += (y_pred_class == y).sum()/len(y)
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    loss_curves['train_loss'].append(train_loss)
    loss_curves['train_acc'].append(train_acc)
    
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
      for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        test_pred_log = model(X)
        loss = loss_fn(test_pred_log, y)
        test_loss += loss
        test_pred_class = torch.softmax(y_pred_log, dim=1).argmax(dim=1)
        test_acc += (test_pred_class == y).sum()/len(y)
      test_loss = test_loss / len(test_dataloader)
      test_acc = test_acc / len(test_dataloader)
      loss_curves['test_loss'].append(test_loss)
      loss_curves['test_acc'].append(test_acc)
    print(f'Epoch: {epoch+1}\n train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}'
    f' | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}')
  return loss_curves

def show_random_images(path, class_name='*', nrows=3, ncols=3, figsize=(12, 8)):
  """Shows random images given path and class name. The dataset must be should be in
  format 'path/*/classname/*.jpg'"""
  images_paths = list(Path(path).glob(f'*/{class_name}/*.jpg'))
  random_paths = random.sample(images_paths, k=nrows*ncols)
  fig = plt.figure(figsize=figsize)
  for i in range(nrows*ncols):
    img_path = random_paths[i] 
    img = Image.open(img_path)
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(img)
    plt.title(f'Image size: {img.size}\nImage class: {img_path.parent.stem}')
    plt.axis(False)
