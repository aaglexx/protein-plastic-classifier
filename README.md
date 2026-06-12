# protein-plastic-classifier

Binary classifier that predicts whether a protein can degrade plastic (PET).

Built with ESM-2 (Meta's protein language model) as a feature extractor and logistic regression as the classifier.

## How it works

```
protein sequence → ESM-2 (embeddings) → logistic regression → plastic / not plastic
```

## Results

Trained on 9212 proteins (4212 positive, 5000 negative) from UniProtKB.

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |

## Usage

```bash
pip install transformers torch biopython scikit-learn
python predict.py
```

```
sequence: MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPF...
plastic (99.97%)
```

## Dataset

- Positive: [UniProtKB petase query](https://www.uniprot.org/uniprotkb?query=petase) — 4212 proteins
- Negative: reviewed proteins with no petase/hydrolase/cutinase activity — 5000 proteins

## Stack

- [ESM-2 35M](https://huggingface.co/facebook/esm2_t12_35M_UR50D) — protein language model by Meta
- scikit-learn — logistic regression classifier
- BioPython — FASTA parsing

## Files

- `predict.py` — inference script
- `plastic_classifier.pkl` — trained classifier
- `notebook.ipynb` — full training pipeline
