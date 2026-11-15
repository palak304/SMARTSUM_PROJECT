import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

print("Loading model for evaluation...")
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
rouge = evaluate.load("rouge")

try:
    df = pd.read_csv("datasets/cleaned_data.csv")
except:
    df = pd.read_csv("datasets/cnn_dailymail.csv")

sample_df = df.head(10)
predictions = []
references = []

print("Generating summaries for evaluation...")
for index, row in sample_df.iterrows():
    article = str(row.get('article', ''))
    reference = str(row.get('highlights', ''))
    
    if len(article.strip()) > 50:
        inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, num_beams=4)
        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        predictions.append(pred)
        references.append(reference)

results = rouge.compute(predictions=predictions, references=references)
print("\n--- ROUGE Evaluation Results ---")
print(results)