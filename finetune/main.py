from services.search import train
import psycopg2
import json
from datetime import datetime
import os
import sys
import pandas as pd



from utils.data import get_conn


def get_unseen_data():

    conn, cur = get_conn()
    cur.execute("""SELECT * 
                FROM user_activity 
                WHERE finetuned = FALSE 
                AND action_type = 'select'""")
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

    if len(records) == 0:
        print("No unseen data found")
        return pd.DataFrame()

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
    print(f"Updated {len(df)} records")
    conn.close()

def finetune():
    df = get_unseen_data()
    if len(df) == 0:
        return
    train(df)
    update_finetuned_status(df)


def reset_finetuned():
    conn, cur = get_conn()
    cur.execute("UPDATE user_activity SET finetuned = FALSE")
    conn.commit()
    print("Reset all rows to finetuned = FALSE")
    conn.close()


if __name__ == "__main__":
    finetune()

