"""Multilingual NLI scorer: does the model's answer still entail the reference answer?"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_MODEL = "joeddav/xlm-roberta-large-xnli"


class NLIScorer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(self.device).eval()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.ent_id = next(i for i, l in labels.items() if l == "entailment")

    @torch.no_grad()
    def score(self, predictions, references, bs=32):
        """premise = model answer, hypothesis = reference answer.
        Returns (entailment probabilities, boolean 'answer still conveys the fact' flags)."""
        probs, flags = [], []
        for i in range(0, len(predictions), bs):
            enc = self.tok(predictions[i : i + bs], references[i : i + bs], truncation=True,
                           max_length=256, padding=True, return_tensors="pt").to(self.device)
            p = self.model(**enc).logits.softmax(-1)
            probs += p[:, self.ent_id].tolist()
            flags += (p.argmax(-1) == self.ent_id).tolist()
        return probs, flags
