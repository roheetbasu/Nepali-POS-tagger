import torch
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np


def compute_metrics(predictions, labels, id2label):
    # Convert predicted logits into label ids
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100] 
        for pred, lab in zip(predictions, labels)
    ]
    
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100] 
        for pred, lab in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_model(model,train_dataloader,test_dataloader,label2ids,epochs=10,lr=5e-5):
    
    #moving model to cuda if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    #optimizer
    optimizer = AdamW(model.parameters(),lr = lr)
    
    #learning rate scheduler
    epochs = epochs
    num_training_steps = epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps = 0,
    num_training_steps = num_training_steps
    )
    
    progress_bar = tqdm(range(num_training_steps))
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_loss = total_loss / len(train_dataloader)
        print(f"Training Loss: {avg_loss:.4f}")

        # ----------------- Evaluation -----------------
        overall_results = {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0, "count": 0}
        
        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            with torch.no_grad():
                output = model(**model_inputs)

            logits = output.logits.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            batch_size = labels.shape[0]

            batch_results = compute_metrics(logits, labels, model.config.id2label)

            # update weighted average
            overall_results["precision"] += batch_results["precision"] * batch_size
            overall_results["recall"] += batch_results["recall"] * batch_size
            overall_results["f1"] += batch_results["f1"] * batch_size
            overall_results["accuracy"] += batch_results["accuracy"] * batch_size
            overall_results["count"] += batch_size

        # final average
        final_results = {k: v / overall_results["count"] for k, v in overall_results.items() if k != "count"}
        print(f"F1: {final_results['f1']:.4f} | Accuracy: {final_results['accuracy']:.4f}")
        
        # save best model
        if final_results['f1'] > best_f1:
            torch.save(model.state_dict(), "Model/bestmodel.pt")
            best_f1 = final_results['f1']
            print(f"✅ New best model saved with F1: {best_f1:.4f}")
        