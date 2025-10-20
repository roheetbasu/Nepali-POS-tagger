from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader


def get_dataloader(tokenizer, data_set, batch_size = 32, shuffle=True):
    #datacollator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    #dataloader for train
    dataloaded = DataLoader(
        data_set,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator
        )
    
    return dataloaded