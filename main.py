import numpy as np
import pandas as pd
from SRC.preprocessing import read_and_clean_data
from SRC.tokenizer import tokenize_and_align_labels,load_tokenizer
from SRC.model import load_model
from SRC.dataloader import get_dataloader
from SRC.train_and_eval import train_model
from SRC.test import test_model
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset


#Read and Clean dataset
file = "data/POS.csv"

data, label2ids, ids2label = read_and_clean_data(file)

train, test = train_test_split(data,test_size=0.2,shuffle=True)

train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)

#convert into Arrow(Hugging Face dataset)
dataset_train = Dataset.from_pandas(train)
dataset_test = Dataset.from_pandas(test)

tokenizer = load_tokenizer("Shushant/nepaliBERT")

#tokenizing dataset
tokenized_train = dataset_train.map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={
        "tokenizer" : tokenizer,
        "label2ids": label2ids
        }
)

tokenized_test = dataset_test.map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={
        "tokenizer" : tokenizer,
        "label2ids": label2ids
        }
)
# Remove non-input columns
tokenized_train = tokenized_train.remove_columns(['words'])
tokenized_test = tokenized_test.remove_columns(['words'])

# Format as torch
tokenized_train.set_format(type="torch")
tokenized_test.set_format(type="torch")

train_dataloader = get_dataloader(tokenizer, tokenized_train, batch_size=32,shuffle=True)
test_dataloader = get_dataloader(tokenizer, tokenized_test, batch_size=32,shuffle=False)

#loading model
model = load_model("Shushant/nepaliBERT",label2ids)

#training model
train_model(model,train_dataloader,test_dataloader,label2ids,epochs=10,lr=5e-5)

#test model 
test_model()




