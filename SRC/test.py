from transformers import AutoTokenizer, AutoModelForTokenClassification
from SRC.tokenizer import prepare_sentences_for_tokenizer
import torch


def test_model():

    # Load tokenizer and model
    model_name_or_path = "roheetbasu/Nepali-pos-tagger"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Nepali test sentences
    sentences = [
        "नेपाल सुन्दर देश हो।",
        "म स्कूल जान्छु।",
        "कुकुर बगैँचामा खेलिरहेको छ।",
        "उसले मलाई भेट्यो।",
        "विद्यालयमा विद्यार्थीहरू पढिरहेका छन्।",
        "म किताब घरमा राखेँ।",
        "बजारबाट तरकारी किनेर ल्याएँ।",
        "तिमी कहाँ जान्छौ?",
        "सूर्य पश्चिमतिर अस्ताउँछ।",
        "हामीले फुटबल खेल्ने योजना बनायौं।"
    ]

    # Get properly tokenized sentences
    split_sentences = prepare_sentences_for_tokenizer(sentences)

    # Tokenize without converting to tensors (needed for word_ids)
    tokenized = tokenizer(
        split_sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128
    )

    # Convert to tensors for model
    inputs = {k: torch.tensor(v).to(device) for k,v in tokenized.items()}

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to predictions
    predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()

    # Map predictions back to words using tokenized.word_ids
    for i, sentence in enumerate(sentences):
        word_ids = tokenized.word_ids(batch_index=i)  # ✅ use tokenized, not inputs
        pred_labels = []
        prev_label = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if idx == 0 or word_idx != word_ids[idx-1]:
                cur_label = model.config.id2label[predictions[i][idx]]
                
                if prev_label == 'VBKO' and cur_label == "PKO":
                    continue
                prev_label = cur_label
                pred_labels.append(cur_label)
                
        print(f"Sentence: {sentence}")
        print(f"POS Tags: {pred_labels}")
        print("---------------")
