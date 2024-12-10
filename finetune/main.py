import pandas as pd
from services.search import train, rebuild_index


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

def save(model, index, dir):
    model.save(dir)
    index.save(dir)
    print(f"Returned model, save to {dir}")
    pass



def trigger_api_reload():
    # Trigger API to reload model and index
    pass

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
    model = train(df)
    new_index = rebuild_index(model)
    save(model, new_index)
    # trigger_api_reload()
    # update_finetuned_status(df)


def reset_finetuned():
    conn, cur = get_conn()
    cur.execute("UPDATE user_activity SET finetuned = FALSE")
    conn.commit()
    print("Reset all rows to finetuned = FALSE")
    conn.close()


if __name__ == "__main__":
    finetune()

