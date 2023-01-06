import sys
import os
import random
import json
import cv2
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict 
from zipfile import ZipFile
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from torch.utils.data.sampler import Sampler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import timm

from data import CustomDataset, BatchSampler, DynamicSampler

def batch_hard_triplet_loss(embeddings, labels, alpha = 0.2):
    distance_matrix = torch.cdist(embeddings, embeddings, p = 2.0)
    distance_matrix = torch.square(distance_matrix)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~(torch.eye(labels.size(0)).bool())).to(embeddings.devic)
    positive_mask = positive_mask.float()
    negative_mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))
    negative_mask = negative_mask.float()
    anchor_positive_dist = positive_mask * distance_matrix
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim = True)

    temp, _ = distance_matrix.max(1, keepdim = True)
    anchor_negative_dist = distance_matrix + temp * (1.0 - negative_mask)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim = True)

    tl = hardest_positive_dist - hardest_negative_dist + alpha
    tl[tl < 1e-9] = 0
    return tl.mean()

def evaluate(model, dataloader, val):
    def extract_embedding(data_loader):
        embeds = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                image_embeddings = model(images.to('cuda'))
                embeds.append(image_embeddings.cpu())

        image_embeddings = np.concatenate(embeds)    
        return image_embeddings

    def predict(image_embeddings, distance_threshold, df, KNN_model):
        preds = []
        CHUNK = 2560
        
        CTS = len(image_embeddings)//CHUNK
        if len(image_embeddings)%CHUNK!=0: CTS += 1

        for j in range(CTS):
            a = j*CHUNK
            b = (j+1)*CHUNK
            b = min(b,len(image_embeddings))
        
            distances, indices = KNN_model.kneighbors(image_embeddings[a:b,])

            for k in range(b-a):
                IDX = np.where(distances[k,]<distance_threshold)[0] # for each embedding-k, find indices of other embedding having distance < threshold, in distances
                IDS = indices[k,IDX]                 # for each embedding-k, find indices of other embedding having distance < threshold, in indices (real indices)
                o = df.iloc[IDS].posting_id.values # get the posting_id of found embedding
                preds.append(o)
        return preds

    # F1-Score
    def getMetric(col):
        def f1score(row):
            n = len( np.intersect1d(row.target,row[col]) )
            return 2*n / (len(row.target)+len(row[col]))
        return f1score

    KNN_classes = 50
    KNN_model = NearestNeighbors(n_neighbors=KNN_classes)

    val_embeddings = extract_embedding(dataloader)
    val_embeddings = normalize(val_embeddings, norm='l2', axis=1)
    KNN_model.fit(val_embeddings)

    preds = predict(val_embeddings, 0.6, val, KNN_model)

    tmp = val.groupby('label_group').posting_id.agg('unique').to_dict()
    val['target'] = val.label_group.map(tmp)
    val['pred'] = preds

    val['f1score'] = val.apply(getMetric('pred'),axis=1)
    f1_score = val['f1score'].mean()
    return f1_score

def main(model_name):
    batch_size = 32
    max_epoch = 50
    lr = 0.01
    device = 'cuda'
    label_per_batch = 4

    data_csv = pd.read_csv('data/train.csv')

    train = pd.read_csv('data/train_batch.csv')
    val = pd.read_csv('data/val_batch.csv')

    X_train, y_train, X_test, y_test = train['image'].tolist(), train['label_group'].tolist(), val['image'].tolist(), val['label_group'].tolist()

    X_train = ['data/train_images/' + x for x in X_train]
    X_test = ['data/train_images/' + x for x in X_test]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale = (0.8, 1.0), interpolation = transforms.InterpolationMode.LANCZOS),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees = 15, interpolation = transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CustomDataset(X_train, y_train, transform = train_transform)
    test_dataset = CustomDataset(X_test, y_test, transform = test_transform, train = False)

    sampler = DynamicSampler(y_train, batch_size, label_per_batch)
    batch_sampler = BatchSampler(sampler, batch_size, True)

    train_dataloader = DataLoader(train_dataset, batch_sampler = batch_sampler, pin_memory = True, num_workers = 0)
    train_val_dataloader = DataLoader(CustomDataset(X_train, y_train, transform = test_transform), batch_size=batch_size, drop_last=False, pin_memory = True, num_workers = 0)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size * 2, drop_last = False, pin_memory = True, num_workers = 0)

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = 0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 25, eta_min=lr*0.01)

    ckpt_dir = 'checkpoints/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
      
    model.eval()
    best_acc = evaluate(model, test_dataloader, val)

    print("Best F1-score: {:.4f}%".format(best_acc * 100))

    last_epoch = 0
    ckpt_path = os.path.join(ckpt_dir, 'lastest_result.pt')
    best_path = os.path.join(ckpt_dir, 'best_result.pt')

    if os.path.exists(best_path):
        ckpt = torch.load(best_path)
        try:
            model.load_state_dict(ckpt['model'])
            best_acc = ckpt['best_acc']
            scheduler.load_state_dict(ckpt['scheduler'])
            last_epoch = ckpt['scheduler']['last_epoch']
            optimizer.load_state_dict(ckpt['optimizer'])
        
        except RuntimeError as e:
            exception_list = ['logits.weight', 'logits.bias']
            old_state_dict = ckpt['model']
            new_state_dict = model.state_dict()
            for key in old_state_dict.keys():
                if key not in exception_list:
                    new_state_dict[key] = old_state_dict[key]
            model.load_state_dict(new_state_dict)

            print('wrong checkpoint')
        
        print('checkpoint is loaded !')
        print("Best F1-score: {:.2f}%".format(best_acc * 100))

    for epoch in range(last_epoch, max_epoch):
        model.train()
        running_loss = []
        train_num_data = 0

        for step, (inputs, labels) in tqdm(enumerate(train_dataloader), desc = 'Training', leave = False, total = len(train_dataloader)):
            inputs = inputs.to(device)
            embeddings = F.normalize(model(inputs), dim = 1, p = 2)
            labels = labels.to(device)

            #triplet loss
            loss = batch_hard_triplet_loss(embeddings, labels, alpha = 0.2)

            optimizer.zero_grad()  
            loss.backward()    
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())    

            train_num_data += inputs.shape[0]

            sys.stdout.write('\r')
            sys.stdout.write("Epoch: {}/{} - Learning rate: {:.6f} - Min/Avg/Max train Loss: {:.6f}/{:.6f}/{:.6f}".format(epoch+1, max_epoch, optimizer.state_dict()['param_groups'][0]['lr'], min(running_loss), np.mean(running_loss), max(running_loss)))
            sys.stdout.flush()

            scheduler.step(epoch + step / len(train_dataloader))
        
        mean_loss = np.mean(running_loss)
        model.eval()
        val_acc = evaluate(model, test_dataloader, val)
        print("\nEpoch: {}/{} - Train Loss: {:.6f}. Val F1-score: {:.4f}%".format(epoch+1, max_epoch, np.mean(running_loss), val_acc * 100))

          
        if best_acc < val_acc:
            best_acc = val_acc
            ckpt = {'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                    'best_acc':best_acc}
            torch.save(ckpt, best_path)
            print('checkpoint is saved!')

        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'best_acc':best_acc}
        torch.save(ckpt, ckpt_path)

if __name__ == "__main__":
    main(sys.argv[1])