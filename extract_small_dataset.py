import json
import pandas as pd

# Load the JSON file
with open('HateXplain_dataset.json', 'r') as file:
    data = json.load(file)

# Prepare lists to hold the extracted data
post_ids = []
input_texts = []
labels = []
targets = []

# Function to get the majority label
def get_majority_label(annotators):
    label_counts = {}
    for annotator in annotators:
        label = annotator['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    return max(label_counts, key=label_counts.get)

# Function to get the majority targets
def get_majority_targets(annotators):
    target_counts = {}
    for annotator in annotators:
        for target in annotator.get('target', []):
            target_counts[target] = target_counts.get(target, 0) + 1
    # Only return targets that appear in more than 50% of the annotations
    total_annotations = len(annotators)
    majority_targets = [target for target, count in target_counts.items() if count / total_annotations > 0.5]
    return majority_targets if majority_targets else ["None"]

# Extract necessary information from the JSON data
for post_id, post_data in data.items():
    post_ids.append(post_data['post_id'])
    input_texts.append(" ".join(post_data['post_tokens']))
    labels.append(get_majority_label(post_data['annotators']))
    targets.append(get_majority_targets(post_data['annotators']))

# Create a DataFrame with the extracted information
df_small = pd.DataFrame({
    'post_id': post_ids,
    'input_text': input_texts,
    'label': labels,
    'target': targets
})

# Normalize the input text for easier matching later
df_small['input_text_normalized'] = df_small['input_text'].str.lower().str.strip()

# Save the DataFrame to a CSV file
df_small.to_csv('small_HateXplain_dataset.csv', index=False)

# Optional: Save to a JSON file instead
df_small.to_json('small_HateXplain_dataset.json', orient='records', lines=True)

print("Extraction complete. Saved to 'small_HateXplain_dataset.csv'.")

from sentence_transformers import SentenceTransformer
import numpy as np

print("Generating Sentence-BERT embeddings (this may take a minute)...")
# Load the embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode all the normalized text from the small dataset
embeddings = embedding_model.encode(df_small['input_text_normalized'].tolist(), show_progress_bar=True)

# Save the numpy array
np.save('precomputed_embeddings.npy', embeddings)
print("Embeddings saved to 'precomputed_embeddings.npy'.")