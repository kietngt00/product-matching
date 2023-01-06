import numpy as np
import pandas as pd
import torch

import cudf, cupy, cuml
from cuml.feature_extraction.text import TfidfVectorizer
import cuml.neighbors

import warnings
warnings.filterwarnings("ignore")
import os
import gc
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print('Good to go!')
else:
    print('Please set GPU via Edit -> Notebook Settings.')

from utils import *
from data import *
from model import *

def train_bert_base(train_df, text_dataloader):

    # Choose model name (currently version multilingual bert)
    model_name = config.text_model_name[2] 

    model = Bert(model_name) 
    bert_embeddings = get_bert_embedding(model, text_dataloader)
    KNN_classes = config.knn
    KNN_model = cuml.neighbors.NearestNeighbors(n_neighbors=KNN_classes)
    bert_embeddings_numpy = np.array(bert_embeddings.cpu())
    KNN_model.fit(bert_embeddings_numpy)

    if model_name == "bert-base-multilingual-cased":
        pred = predict_knn(bert_embeddings_numpy, 5, train_df, KNN_model)

    if model_name == "sentence-transformers/all-mpnet-base-v2":
        pred = predict_knn(bert_embeddings_numpy, 1.7, train_df, KNN_model)

    if model_name == "distilbert-base-multilingual-cased":
        pred = predict_knn(bert_embeddings_numpy, 3.6, train_df, KNN_model)
    
    train_df["bert_pred"] = pred
    
    torch.cuda.empty_cache()
    return train_df

def train_tfidf(train_df_cu, train_df):
    model = TfidfVectorizer(stop_words=None, binary=True, max_features =22500)
    tfidf_embeddings = model.fit_transform(train_df_cu.title).toarray()
    KNN_classes = config.knn
    KNN_model_tfidf = cuml.neighbors.NearestNeighbors(n_neighbors=KNN_classes)
    KNN_model_tfidf.fit(tfidf_embeddings)
    pred2 = predict_knn(cupy.asnumpy(tfidf_embeddings), 0.8, train_df, KNN_model_tfidf)
    train_df["tfidf_pred"] = pred2
    torch.cuda.empty_cache()
    return train_df

def combine(train_df):
    tmp = train_df.groupby('image_phash').posting_id.agg('unique').to_dict()
    train_df['oof'] = train_df.image_phash.map(tmp)
    train_df['combine_pred'] = train_df.apply(combine_predictions, axis = 1)
    return train_df


def evaluate(train_df, name='tfidf_pred'):
    """
    name: {tfidf_pred, bert_pred, combine_pred}
    return f1_score
    """

    train_df['f1score'] = train_df.apply(getMetric(name),axis=1)
    f1_score = train_df['f1score'].mean()
    print('Mean f1 score: ', f1_score)
    return f1_score

def run(experiment="tfidf"):
    test = pd.read_csv(config.PATH + 'test.csv')
    if len(test) > 3:
        TRAIN = False
    else:
        TRAIN = True
    if TRAIN:
        df = read_dataset("train")
        label_group_dict = df.groupby("label_group").posting_id.agg("unique").to_dict()
        df["target"] = df.label_group.map(label_group_dict)
    else:
        read_dataset("test")

    df_cu = cudf.DataFrame(df)
    df_prep = preprocess_df(df_cu,'title', filters=punctuation)
    df_text = ShopeeTextDataset(df)
    text_dataloader = torch.utils.data.DataLoader(df_text, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    if (experiment=="tfidf"):
        final_df = train_tfidf(df_cu, df)
        evaluate(final_df)

    elif (experiment=="bert_base"):
        final_df = train_bert_base(df, text_dataloader)
        evaluate(final_df, name='bert_pred')

    elif (experiment == "combine"):
        df = train_tfidf(df_cu, df)
        final_df = combine(df)
        evaluate(final_df, name='combine_pred')


