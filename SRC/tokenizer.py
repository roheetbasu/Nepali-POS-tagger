from transformers import AutoTokenizer

def load_tokenizer(tokenizer_name):
    #importing tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def tokenize_and_align_labels(tokenizer, examples, label2ids):

    tokenize_inputs = tokenizer(examples['words'],truncation=True,max_length = 512,is_split_into_words=True)
    labels = []

    for i,label in enumerate(examples['labels']):
        word_ids = tokenize_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2ids[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenize_inputs['labels'] = labels
    return tokenize_inputs
                
        
    