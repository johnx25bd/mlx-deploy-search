from services.search import train
import psycopg2
import json
from datetime import datetime
import os
import pandas as pd

def get_conn():
    # Use environment variables for flexibility between local and Docker
    host = os.getenv('DB_HOST', 'localhost')  # defaults to localhost for local dev
    port = os.getenv('DB_PORT', '5433')       # use 5433 since that's what we mapped in Docker

    conn = psycopg2.connect( # TODO: Read in credentials from env
            dbname="user_logs",
            user="logger",
            password="secure_password",
            host=host,
            port=port
        )
    cur = conn.cursor()
    return conn, cur

def get_unseen_data():

    conn, cur = get_conn()
    cur.execute("SELECT * FROM user_activity WHERE finetuned = FALSE AND action_type = 'select'")
    records = cur.fetchall()
    conn.close()

    columns = [
        'id',
        'user_id',
        'action_type',
        'action_details',
        'ip_address',
        'timestamp',
        'finetuned'
    ]

    df = pd.DataFrame(records, columns=columns)
    df['query'] = df['action_details'].apply(lambda x: x['query'])
    df['doc_relevant_id'] = df['action_details'].apply(lambda x: x['results'])

    return df

def update_finetuned_status(df):
    conn, cur = get_conn()
    cur.execute("""UPDATE user_activity 
                SET finetuned = TRUE 
                WHERE id IN ({})"""
                .format(','.join(df['id'].astype(str))))
    conn.commit()
    conn.close()

def finetune():
    df = get_unseen_data()
    train(df)
    # update_finetuned_status(df)



if __name__ == "__main__":
    finetune()

