import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pypdf

st.set_page_config(page_title="Summarix - Abstractive Summarizer", page_icon="📝", layout="wide")

st.title("📝 Summarix: Abstractive Text Summarisation")
st.markdown("Generate concise, human-like summaries from long articles or documents using Transformer models.")

@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading NLP Model... Please wait."):
    tokenizer, model = load_model()

st.sidebar.header("⚙️ Settings")
length_option = st.sidebar.selectbox("Select Summary Length:", ("Short", "Medium", "Detailed"))

if length_option == "Short":
    min_len, max_len = 25, 50
elif length_option == "Medium":
    min_len, max_len = 50, 100
else:
    min_len, max_len = 100, 180

input_text = st.text_area("Paste your long article or text here:", height=250)

if st.button("✨ Generate Summary", type="primary"):
    if len(input_text.strip()) < 50:
        st.warning("⚠️ Please enter at least 50 characters.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(inputs["input_ids"], max_length=max_len, min_length=min_len, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("📌 Generated Summary")
        st.success(summary)
        
        col1, col2 = st.columns(2)
        col1.metric("Original Words", len(input_text.split()))
        col2.metric("Summary Words", len(summary.split()))