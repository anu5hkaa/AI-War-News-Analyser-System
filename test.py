from transformers import AutoModelForSequenceClassification

print("Before model")

model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli"
)

print("Model loaded")