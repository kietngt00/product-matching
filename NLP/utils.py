import numpy as np
import torch
import cudf
import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
    
def cosine_dist(x,y):
    m, n = x.size(0), y.size(0)
    norm_x = x.norm(p=2, dim=1, keepdim=True).expand(m,n)
    norm_y = y.norm(p=2, dim=1, keepdim=True).expand(n,m).T
    dist = torch.matmul(x, y.T) / (norm_x*norm_y)
    return dist
    
def predict_knn(text_embeddings, distance_threshold, df, KNN_model):
    preds = []
    CHUNK = 2560
    print(len(df))
    CTS = len(text_embeddings)//CHUNK
    if len(text_embeddings)%CHUNK!=0: CTS += 1
    for j in range(CTS):
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(text_embeddings))
        print('chunk',a,'to',b)
        distances, indices = KNN_model.kneighbors(text_embeddings[a:b,])
        for k in range(b-a):
            IDX = np.where(distances[k,]<distance_threshold)[0] 
            IDS = indices[k,IDX]               
            o = df.iloc[IDS].posting_id.values
            preds.append(o)
    return preds

def predict_cosine(embeddings, df, threshold=0.85, least_threshold=65, k=50):
    predict = []
    CHUNK = 1024
    n = (embeddings.size(0) + CHUNK - 1) // CHUNK
    with torch.no_grad():
        for i in range(n):
            a = i*CHUNK
            b = min((i+1)*CHUNK, embeddings.size(0))
            x = embeddings[a:b]
            y = embeddings
            chunk_distance = cosine_dist(x,y)
            topK = torch.topk(chunk_distance, k=min(k, embeddings.size(0)))
            topK_idx, topK_dist = topK[1].detach().cpu().numpy(), topK[0].detach().cpu().numpy() # size: [chunk, k]

            for j, (idx, dist) in enumerate(zip(topK_idx, topK_dist)):
                mask = dist >= threshold
                # release threshold if match < 2
                if least_threshold is not None and not mask[1]:
                    mask[1] = True if dist[1] >= least_threshold else False
                target_index = idx[mask]
                pred = df.iloc[target_index].posting_id.to_numpy()
                predict.append(pred)
    return predict

# F1-Score
def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

# Function to combine predictions of TFIDF vs image_phash
def combine_predictions(row):
    x = np.concatenate([row['tfidf_pred'], row["oof"]])
    return np.unique(x)
