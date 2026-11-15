# SmartSum – Abstractive Text Summarisation

## Project Overview
SmartSum is an NLP-based project that automatically generates short and meaningful summaries from long articles using Transformer-based abstractive text summarisation.

## Objectives
- To reduce the length of lengthy articles.
- To generate meaningful and readable summaries.
- To save users' time.
- To use Natural Language Processing and Transformer models.

## Technologies Used
- Python
- Natural Language Processing (NLP)
- Transformers
- PyTorch
- Hugging Face
- Streamlit
- Pandas

## Dataset
CNN/DailyMail dataset is used for training and evaluation.

## Project Features
- Enter or paste a long article.
- Generate an automatic summary.
- Display the generated summary.
- Simple and user-friendly interface.

## Project Structure
```text
SmartSum_Project/
├── datasets/
├── models/
├── screenshots/
├── app.py
├── train.py
├── preprocess.py
├── evaluate.py
├── download_dataset.py
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Run the Project
```bash
streamlit run app.py
```

## Future Scope
- Support multiple languages.
- Improve summary accuracy.
- Add PDF and document upload.
- Provide different summary lengths.

## Author
Palak
