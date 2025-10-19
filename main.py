import numpy as np
import pandas as pd
from SRC.preprocessing import read_and_clean_data
from sklearn.model_selection import train_test_split

from datasets import Dataset


#Read and Clean dataset
file = "data/POS.csv"

data, label2ids, ids2label = read_and_clean_data(file)

train, test = train_test_split(data,test_size=0.2,shuffle=True)

train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)

dataset_train = Dataset.from_pandas(train)
dataset_test = Dataset.from_pandas(test)