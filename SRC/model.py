from transformers import AutoModelForTokenClassification


def load_model(model_name,label2ids):
    NUM_LABELS = len(label2ids)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        label2id = label2ids,
    )
    return model


    

    
    
    
