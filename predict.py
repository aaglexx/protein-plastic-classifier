
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = model.to(device)
model.eval()

with open("plastic_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

def get_embedding(sequence, max_length=512):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def predict(sequence):
    emb = get_embedding(sequence)
    proba = clf.predict_proba([emb])[0][1]
    label = clf.predict([emb])[0]
    result = "plastic" if label == 1 else "not plastic"
    print(f"{result} ({proba:.2%})")
    return label, proba

if __name__ == "__main__":
    seq = input("sequence: ")
    predict(seq)
