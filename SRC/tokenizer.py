from transformers import AutoTokenizer
import re

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
                

def tokenize_sentence_for_model(sentence):
    """
    Tokenize a Nepali sentence into words suitable for HuggingFace tokenizers.
    - Separates known particles
    - Splits punctuation
    Returns a list of words (tokens)
    """
    sentence = re.sub(r'([।,!?])', r' \1 ', sentence)
    tokens = sentence.strip().split()
    
    # List of Nepali particles
    particles = ["को", "का", "की", "मा", "बाट", "लाई", "देखि", "सम्म", "सँग", 
    "भित्र", "बाहिर", "माथि", "तल", "पछाडि", "अगाडि", "तर्फ", 
    "वरिपरि", "भर", "बीच", "जस्तो", "अनुसार", "अन्तर्गत", "प्रति",'ले']

    final_tokens = []
    for token in tokens:
        matched = False
        for p in particles:
            if token.endswith(p) and len(token) > len(p):
                base = token[:-len(p)]
                final_tokens.append(base)
                final_tokens.append(p)
                matched = True
                break
        if not matched:
            final_tokens.append(token)

    return final_tokens


def prepare_sentences_for_tokenizer(sentences):
    """
    Takes a list of Nepali sentences and returns a list of token lists
    ready for tokenizer with is_split_into_words=True
    """
    return [tokenize_sentence_for_model(s) for s in sentences]

