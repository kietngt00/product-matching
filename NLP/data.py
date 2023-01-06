import pandas as pd

import nltk
import cudf

from torch.utils.data import Dataset
from model import config

import warnings
warnings.filterwarnings("ignore")


class ShopeeTextDataset(Dataset):
    def __init__(self, df):
        super(ShopeeTextDataset, self).__init__()
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        title = self.df.title.iloc[index]
        return title



def read_dataset(name="train"):
    assert name in {"train", "test"}
    df = pd.read_csv(config.PATH + f'{name}.csv')
    return df



class ShopeeTextDataset(Dataset):
    def __init__(self, df):
        super(ShopeeTextDataset, self).__init__()
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        title = self.df.title.iloc[index]
        return title


STOPWORDS = nltk.corpus.stopwords.words('english')

punctuation = [ '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/',  '\\', ':', ';', '<', '=', '>',
           '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '\t','\n',"'",",",'~' , '—']

def text_preprocessing(input_text, filters=None, stopwords=STOPWORDS):
    # filter punctuation 
    translation_table = {ord(char): ord(' ') for char in filters}
    input_text = input_text.str.translate(translation_table)
    #convert to lower case
    input_text = input_text.str.lower()
    # remove stopwords 
    stopwords_gpu = cudf.Series(stopwords)
    input_text =  input_text.str.replace_tokens(stopwords_gpu, ' ')
    # normalize spaces
    input_text = input_text.str.normalize_spaces( )
    # strip leading and trailing spaces
    input_text = input_text.str.strip(' ')
    return input_text

def preprocess_df(df, col, **kwargs):
    df[col] = text_preprocessing(df[col], **kwargs)
    return  df
