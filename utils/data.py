import os
import torch
import gensim
import psycopg2
import numpy as np
import pandas as pd
import gensim.downloader as api
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.core import DocDataset, collate_docdataset


def collate(batch):

    try:
        docs_rel, docs_irr, queries = zip(*batch)
        # print(f"docs_rel shape: {docs_rel[0].shape}")
        # print(f"docs_irr shape: {docs_irr[0].shape}")
        # print(f"queries shape: {queries[0].shape}")

        docs_rel = pad_sequence(docs_rel, batch_first=True, padding_value=0)
        docs_irr = pad_sequence(docs_irr, batch_first=True, padding_value=0)
        queries = pad_sequence(queries, batch_first=True, padding_value=0)

        # print(f"docs_rel shape: {docs_rel.shape}")
        # print(f"docs_irr shape: {docs_irr.shape}")
        # print(f"queries shape: {queries.shape}")

        # Create masks one by one
        # print("Creating docs_rel_mask...")
        docs_rel_mask = (docs_rel != 0).float()
        # print("docs_rel_mask created successfully")

        # print("Creating docs_irr_mask...")
        docs_irr_mask = (docs_irr != 0).float()
        # print("docs_irr_mask created successfully")

        # print("Creating query_mask...")
        query_mask = (queries != 0).float()
        # print("query_mask created successfully")

        return docs_rel, docs_irr, queries, docs_rel_mask, docs_irr_mask, query_mask
    except Exception as e:
        print(f"Error in collate function: {e}")
        print(f"Batch contents: {batch}")
        raise


def get_conn():
    # Use environment variables for flexibility between local and Docker
    host = os.getenv('DB_HOST', 'postgres')  # defaults to localhost for local dev
    port = os.getenv('DB_PORT', '5432')       # use 5433 since that's what we mapped in Docker

    conn = psycopg2.connect( # TODO: Read in credentials from env
            dbname="user_logs",
            user="logger",
            password="secure_password",
            host=host,
            port=port
        )
    cur = conn.cursor()
    return conn, cur


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
