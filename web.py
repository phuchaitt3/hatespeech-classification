import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Cache the tokenizer and model loading
@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move model to the appropriate device
    return tokenizer, model

# Cache the FAISS index creation to avoid recomputing embeddings
@st.cache_resource
def create_faiss_index(_dataset_embeddings):
    d = _dataset_embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance index
    index.add(np.array(_dataset_embeddings))  # Add dataset embeddings to the index
    return index

# Cache Sentence-BERT model loading
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the model and tokenizer with caching
model_path = './final_fine_tuned_bert_2_class'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and embedding model
tokenizer, model = load_model_and_tokenizer(model_path)
embedding_model = load_embedding_model()

# Load your DataFrame
df = pd.read_csv('./small_HateXplain_dataset.csv')  # Replace with your actual DataFrame path

# Normalize the input text for matching
df['input_text_normalized'] = df['input_text'].str.lower().str.strip()

# Load precomputed embeddings
dataset_embeddings = np.load('precomputed_embeddings.npy')  # Load precomputed embeddings

# Fit the LabelEncoder on the labels
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

# Create FAISS index for fast cosine similarity search with caching
index = create_faiss_index(dataset_embeddings)

# Set the model to evaluation mode
model.eval()

def explain_prediction(text, model, tokenizer, df, label_encoder, index, embedding_model, device, num_features, find_target=False, run_lime=True):
    """Generate explanation for a given text using LIME and include the target of hate speech if applicable."""

    if not text.strip():  # Check if the input text is empty
        st.error("Input text cannot be empty. Please enter some text.")
        return

    predicted_label_name = None
    exp = None  # Initialize exp for LIME explanation
    
    if run_lime:
        explainer = LimeTextExplainer(class_names=label_encoder.classes_)

        def model_predict(texts):
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}  
            with torch.no_grad():
                logits = model(**inputs).logits
            return F.softmax(logits, dim=1).cpu().numpy()

        # Run the LIME explanation and store the results
        exp = explainer.explain_instance(
            text, 
            model_predict,
            num_features=num_features,  # Use user-specified number of features
            top_labels=1  
        )

        predicted_label = exp.top_labels[0]
        predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
    else:
        # If LIME is not selected, only predict without explanation
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
        predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
    
    st.write(f"Predicted Label: {predicted_label_name}")

    # Optional: Find the target of hate speech if enabled by the user
    if find_target and predicted_label_name is not None and predicted_label_name != 'normal':
        # Step 1: Split the input text into individual sentences
        input_sentences = text.split('.')

        # Step 2: Generate embeddings for each sentence
        sentence_embeddings = [embedding_model.encode(sentence) for sentence in input_sentences]

        # Step 3: Average the embeddings of all sentences to create a single combined embedding
        combined_embedding = np.mean(np.array(sentence_embeddings), axis=0)

        # Step 4: Use FAISS to search for the most similar text in the dataset
        combined_embedding = np.expand_dims(combined_embedding, axis=0)  # reshape for FAISS
        distances, indices = index.search(combined_embedding, k=1)  # Search in FAISS index

        # Step 5: Get the target group for the most similar text
        most_similar_index = indices[0][0]
        target_group = df.iloc[most_similar_index]['target']

        # Calculate confidence score based on distance
        distance = distances[0][0]
        max_distance = 100  # Reasonable upper bound for distance normalization
        confidence = max(0, 1 - (distance / max_distance))  # Ensure confidence is not negative

        st.write(f"Targets: {target_group}")
        st.write(f"Confidence Score of Targets (0 to 1): {confidence:.2f}")
    
    # Show LIME explanation after the target (if LIME was enabled)
    if run_lime and exp:
        exp_html = exp.as_html()
        st.components.v1.html(exp_html, height=400)


# Define Streamlit app
st.markdown("<h1 style='text-align: center;'>Hate Speech Classifier</h1>", unsafe_allow_html=True)

# Get input text from the user
user_input = st.text_area("Enter the text you want to classify:")

# Only initialize widgets if not set
if 'num_features' not in st.session_state:
    st.session_state.num_features = 3
if 'find_target' not in st.session_state:
    st.session_state.find_target = False
if 'run_lime' not in st.session_state:
    st.session_state.run_lime = True  # Default is to run LIME

# Toggle to include/exclude LIME explanation
st.session_state.run_lime = st.checkbox("Include LIME explanation", value=st.session_state.run_lime)

# Slider for selecting the number of features for LIME explanation
st.session_state.num_features = st.slider("Select the number of words for LIME explanation (if enabled)", 
                                          min_value=1, max_value=10, value=st.session_state.num_features)

# Checkbox for the user to opt-in for finding the target group
st.session_state.find_target = st.checkbox("Find the target of hate speech", value=st.session_state.find_target)

# Button to classify the input
classify_button_clicked = st.button("Classify")  # Only triggers when explicitly clicked

# Only execute classification logic if the button is clicked
if classify_button_clicked:
    if not user_input.strip():
        st.error("Input text cannot be empty. Please enter some text.")
    else:
        with st.spinner('Processing... Please wait.'):
            explain_prediction(
                user_input,
                model,
                tokenizer,
                df,
                label_encoder,
                index,
                embedding_model,
                device,
                st.session_state.num_features,
                st.session_state.find_target,
                st.session_state.run_lime
            )
else:
    st.write("Waiting for user input...")
