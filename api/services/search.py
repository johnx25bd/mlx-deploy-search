

import torch
import torch.nn as nn
import pandas as pd

import faiss

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_embeddings
from utils.text import str_to_tokens
from models.core import DocumentDataset, TwoTowerModel, loss_fn
from models.HYPERPARAMETERS import FREEZE_EMBEDDINGS, PROJECTION_DIM, MARGIN

# Add this near the top of the file
def get_data_path():
    """Return the correct data path whether running locally or in container"""
    # When running in container, files will be in /data
    if os.path.exists('/app/data'):
        return '/app/data'
    # When running locally, look for data in parent directory
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'api', 'data')


data_path = get_data_path()

vector_embeddings_path = os.path.join(data_path, 'word-vector-embeddings.model')
training_with_tokens_path = os.path.join(data_path, 'training-with-tokens.parquet')
doc_index_path = os.path.join(data_path, 'doc-index-64.faiss')
two_tower_state_dict_path = os.path.join(data_path, 'two_tower_state_dict.pth')


vocab, embeddings, word_to_idx = load_embeddings(vector_embeddings_path)
embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=FREEZE_EMBEDDINGS)

EMBEDDING_DIM = embeddings.shape[1]
VOCAB_SIZE = len(vocab)

# TODO: we'll load these from MinIO?
df = pd.read_parquet(training_with_tokens_path)
index = faiss.read_index(doc_index_path)

model = TwoTowerModel(
    embedding_dim=EMBEDDING_DIM,
    projection_dim=PROJECTION_DIM,
    embedding_layer=embedding_layer,
    margin=MARGIN
)


model.load_state_dict(
    torch.load(
        two_tower_state_dict_path,
        weights_only=True,  
        map_location=torch.device('cpu')  # Add this to ensure CPU loading
    )
)

# Function to get nearest neighbors
def get_nearest_neighbors(query, model, df, k=5, index=index, word_to_idx=''):
    query_tokens = torch.tensor([str_to_tokens(query, word_to_idx)])
    query_mask = (query_tokens != 0).float()

    query_encoding = model.query_encode(query_tokens, query_mask)
    query_projection = model.query_project(query_encoding)

    query_vector = query_projection.detach().numpy()
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    indices = indices.squeeze()
    distances = distances.squeeze()
    documents = df.loc[indices]['doc_relevant']
    urls = df.loc[indices]['url_relevant']
    print("get_docs output", documents.to_list(), urls.to_list(), distances.tolist(), indices.tolist())
    return documents.to_list(), urls.to_list(), distances.tolist(), indices.tolist()

def get_docs(q):
    return get_nearest_neighbors(q, model, df, 5, index, word_to_idx)

if __name__ == "__main__":
    print('Getting docs...')
    docs,urls,distances, indices=get_docs("What is the capital of France?")
    print("docs", docs)
    print("urls", urls)
    print("distances", distances)
    print("indices", indices)