from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel

def tokenize_input(text1, text2, label, tokenizer):
    # Ensures that questions and paired, accompanied with label
    tokenized_text = tokenizer(text=[text1, text2], is_split_into_words=True)
    tokenized_text['label'] = label
    return tokenized_text

'''
Custom scoring function for LoRA fine tuning
'''
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro')
    }

class QuantizedLoraPredictor:
    def __init__(self, model_name, peft_model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = PeftModel.from_pretrained(base_model, peft_model_id).merge_and_unload()
        
    def predict(self, text1, text2):
        inputs = self.tokenizer(
            text=[text1, text2],
            is_split_into_words=True,
            return_tensors="pt"
        )  
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()