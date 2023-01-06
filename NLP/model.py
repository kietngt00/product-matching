
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda:0')
cpu = torch.device('cpu')

class config:
    PATH = "../input/shopee-product-matching/" 
    text_model_name = ["distilbert-base-multilingual-cased", 
                       "cahya/bert-base-indonesian-1.5G",
                       'bert-base-multilingual-cased', 
                       'sentence-transformers/all-mpnet-base-v2',
                       'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                    ]
    chkpnt = "../input/bert-text/bert.pt"
    batch_size = 16
    knn = 50

class Bert(nn.Module):
    def __init__(self, model_name, fc_dim=1024):
        super(Bert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(device)

    def forward(self, text):
        inputs = self.tokenizer(text, max_length=100, truncation=True, padding='max_length', return_tensors="pt")
        outputs = self.backbone(input_ids = inputs["input_ids"].to(device), attention_mask = inputs["attention_mask"].to(device))
        embeddings = outputs.last_hidden_state
        masks = inputs["attention_mask"].to(device) # 16*100
        return embeddings, masks


def get_bert_embedding(model, dataloader):
    text_embeddings = []
    with torch.no_grad():
        for text in tqdm(dataloader):
            text = list(text)
            embeddings, attention_mask = model(text) # shape 16 x 100 x 768
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask # apply attention mask
            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            """
             Calculate the mean as the sum of the embedding activations summed divided by the number of values that should be given attention in each position summed_mask
            """
            mean_pooled = summed / summed_mask
            text_embeddings.append(mean_pooled)
            del text
    text_embeddings = torch.cat(text_embeddings, axis=0)
    torch.cuda.empty_cache()   
    return text_embeddings


class RoBerta(nn.Module):
    def __init__(self, fc_dim=1024):
        super(RoBerta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-roberta-large')
        self.backbone = AutoModel.from_pretrained('sentence-transformers/nli-roberta-large')

    def forward(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.backbone(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
        
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_robert_embedding(model, dataloader):
    text_embeddings = []
    with torch.no_grad():
        for text in tqdm(dataloader):
            text = list(text)
            text_embeddings.append(model(text))
            del text
    text_embeddings = torch.cat(text_embeddings, axis=0)
    torch.cuda.empty_cache()   
    return text_embeddings
    
    