import os
import torch
import gensim
import numpy as np
import pandas as pd
import gensim.downloader as api


def load_embeddings(embeddings_path):
    try:
        w2v = gensim.models.Word2Vec.load(embeddings_path)
    except Exception as e:
        print(f"Error loading word2vec model: {e}")
        raise e
    
    vocab = w2v.wv.index_to_key
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    embeddings_array = np.array([w2v.wv[word] for word in vocab])
    embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
    return vocab, embeddings, word_to_idx


def load_word2vec(random_seed=42, embeddings_path='../data/word-vector-embeddings.model', save_path='../data/word-vector-embeddings.model'):

    from utils.text import preprocess_list as preprocess

    pd.set_option('mode.chained_assignment', None)  # Suppress SettingWithCopyWarning
    np.random.seed(random_seed)

    if os.path.exists(embeddings_path):
        try:
            w2v = gensim.models.Word2Vec.load(embeddings_path)
        except Exception as e:
            print(f"Error loading word2vec model: {e}")
            raise e
    else:
        # Train word2vec model
        raw_corpus = api.load('text8')
        corpus = [preprocess(doc) for doc in raw_corpus]
        w2v = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=3, workers=4)
        if save_path:
            w2v.save(save_path)

    # Load the word2vec model, extract embeddings, convert to torch tensor
    # w2v = gensim.models.Word2Vec.load('./word2vec/word2vec-gensim-text8-custom-preprocess.model')
    vocab = w2v.wv.index_to_key
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    embeddings_array = np.array([w2v.wv[word] for word in vocab])
    embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
    return vocab, embeddings, word_to_idx

def log_event(event_type, query, docs):
    import json
    import psycopg2
    from datetime import datetime

    print(f'Logging {event_type} event')
    try:
        # Pull this out into a function
        # And read in credentials from env
        conn = psycopg2.connect( # TODO: Read in credentials from env
            dbname="user_logs",
            user="logger",
            password="secure_password",
            host="postgres"
        )

        user_id = 1
        ip_address = "127.0.0.1"
        timestamp = datetime.now()
        
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_activity (
                user_id,
                action_type,
                action_details,
                ip_address,
                timestamp,
                finetuned
            ) VALUES (
                %s,%s,%s,%s,%s, %s
            )
            """,
            (user_id, event_type, json.dumps({"query": query, "results": docs}), ip_address, timestamp, False)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f'logged {event_type} activity')
    except Exception as e:
        print(f"Failed to log {event_type} activity: {str(e)}")




if __name__ == '__main__':
    vocab, embeddings, word_to_idx = load_word2vec()
    print(embeddings.shape)
