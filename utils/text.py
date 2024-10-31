import re
import torch
import string
import pandas as pd

import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.data.find('corpora/stopwords')

from gensim.utils import simple_preprocess

from utils.data import load_word2vec


# Initialize stemmer and stopwords
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))



def preprocess_list(tokens: list[str]) -> list[str]:
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    tokens = ["[S]"] + tokens + ["[E]"]
    return tokens

def preprocess_query(query: str) -> list[str]:
    if query is None or pd.isna(query):
        return []

    query = query.lower()
    query = re.sub(f"[{string.punctuation}]", "", query)
    tokens = simple_preprocess(
        query, deacc=True
    )  # deacc=True removes accents and punctuations

    tokens = preprocess_list(tokens)
    return tokens

# def str_to_list(s: str) -> list[str]:
#     return preprocess_query(s)

def str_to_tokens(s: str, word_to_idx: dict[str, int]) -> list[int]:
    split = preprocess_query(s)
    return [word_to_idx[word] for word in split if word in word_to_idx]

# TODO: Probably can remove
def tokenize_df(df, word_to_idx):
    # Tokenize all columns in one go
    def tokenize_row(row):
        return (
            str_to_tokens(row['doc_relevant'], word_to_idx),
            str_to_tokens(row['doc_irrelevant'], word_to_idx),
            str_to_tokens(row['query'], word_to_idx)
        )
    
    df[['doc_rel_tokens', 'doc_irr_tokens', 'query_tokens']] = df.apply(tokenize_row, axis=1, result_type='expand')
    return df

def get_string_embedding(input_string, vocab, word_to_idx, embeddings):

    processed_words = preprocess_query(input_string)
    word_embeddings = []

    for word in processed_words:
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            word_embedding = embeddings[word_idx]
            word_embeddings.append(word_embedding)

    if not word_embeddings:
        return torch.zeros(embeddings.shape[1])

    string_embedding = torch.stack(word_embeddings).mean(dim=0)
    return string_embedding

def preprocess(query: str):

    # TODO: This needs to not be loaded every time
    vocab, embeddings, word_to_idx = load_word2vec()

    embedding = get_string_embedding(query, vocab, word_to_idx, embeddings)
    return embedding

__all__ = ['preprocess_list', 'str_to_tokens', 'preprocess_query', 'simple_preprocess', 'str_to_list', 'get_string_embedding', 'preprocess']