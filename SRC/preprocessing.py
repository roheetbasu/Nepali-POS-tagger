import pandas as pd

def read_and_clean_data(file_name):
    #reading the csv file
    data =  pd.read_csv(file_name)

    label_list = data['labels'].unique().tolist()
    data = data.dropna().reset_index(drop=True)

    
    label2ids = {label:i for i,label in enumerate(label_list)}
    ids2label = {values:keys for keys,values in label2ids.items()}
    
    grouped = data.groupby("sentence_id")
    
    df_grouped = grouped.agg({
    "words":lambda x:list(x),
    "labels":lambda x:list(x)
    })
    
    return df_grouped,label2ids,ids2label
    