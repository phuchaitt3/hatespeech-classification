**Transformer-based NLP Hate Speech Classifier**

This repository contains a comprehensive web-based Natural Language Processing (NLP) application designed to detect and classify hate speech. Built upon the Transformer architecture, the model categorizes input text into three distinct classes: **Normal, Hate Speech, or Offensive**. 

Beyond basic classification, the system identifies the likely target community of the hate speech and provides visual, word-level explainability using LIME (Local Interpretable Model-agnostic Explanations).

**Project Features**
*   **3-Class Text Classification:** Distinguishes between explicitly hateful, generally offensive, and normal text.
*   **Target Community Identification:** Utilizes Sentence-BERT and FAISS nearest-neighbor search to identify the specific demographic targeted by the text (e.g., African, Islam, Jewish, Homosexual, Women, etc.).
*   **Interactive Explainability:** Integrates LIME directly into the Streamlit UI to highlight the specific tokens/words that drove the model's prediction.

**Dataset & Preprocessing**
The model is trained on the **HateXplain** benchmark dataset, a comprehensive corpus annotated from multiple perspectives. The raw dataset (`HateXplain_dataset.json`) includes over 20,000 samples.

To prepare this raw JSON data for the web application's target lookup feature, a custom pipeline (`extract_small_dataset.py`) was engineered to execute the following:
*   **Majority Voting Labeling:** Each post is annotated by multiple individuals. The script extracts the final class label based on majority consensus. 
*   **Target Community Extraction:** The script iterates through annotator targets and only assigns a target community if it appears in >50% of the annotations, otherwise defaulting to "None".
*   **Token Aggregation:** Raw token lists are joined and normalized into unified text strings for easier nearest-neighbor matching.
*   **Final Output:** This script outputs `small_HateXplain_dataset.csv`, a lightweight reference table essential for the application's local FAISS similarity search.

*(Note: The rationale extraction and MLM masking were handled separately during the initial model training phase).*

**Model Architecture & Two-Phase Training**

Instead of a simple out-of-the-box fine-tuning process, this project utilizes a highly specialized **Two-Phase Training Pipeline** based on the `BertForMaskedLM` and `BertForSequenceClassification` architectures.

**Phase 1: Pre-Finetuning with BERT Masked Language Model (MLM)**
To force the model to learn domain-specific nuances, the base BERT model undergoes an initial MLM pre-finetuning phase. Crucially, the masking is not random; it is guided by the human-annotated **rationales**. By masking the specific words that humans deemed "hateful," the model's self-attention mechanism is mathematically forced to focus on the context surrounding contextually significant tokens. 

**Phase 2: Final Fine-Tuning for Sequence Classification**
The rationale-aware, pre-finetuned model weights are then transferred to a sequence classification head. The model is trained on the 3-class dataset using an AdamW optimizer (Weight Decay = 0.01) and built-in dropout layers for regularization. An Early Stopping mechanism (Patience = 3, Min Delta = 0.001) was implemented to prevent overfitting based on validation loss.

**Evaluation & Results**

The two-phase rationale-masking approach demonstrated superior performance in distinguishing between classes compared to standard baselines.

*   **AUROC:** 0.858 (Highest overall ability to rank positive instances across classes)
*   **Accuracy:** 0.700 (Correctly classifies 7 out of 10 instances)
*   **Macro F1-Score:** 0.684
*   **Recall (Hate Speech):** 0.830 (Highly effective at ensuring severe hate speech is not missed)

**Model Comparison against HateXplain Baselines:**
*   BiRNN-HateXplain [LIME]: AUROC 0.843 | Accuracy 0.629
*   Standard BERT-HateXplain [Attn/LIME]: AUROC 0.851 | Accuracy 0.698
*   **Our Model (BERT MLM + Classifier + LIME): AUROC 0.858 | Accuracy 0.700**

**Running the Web Application Locally**

The frontend is built using Streamlit. To run this project on your local machine:

1. Clone the repository.
2. Install the necessary dependencies (PyTorch, Transformers, Streamlit, LIME, Sentence-Transformers, FAISS, Pandas, Scikit-learn).
3. **Important Note on Large Files:** Due to file size limits, two dependency files required for the FAISS target-lookup logic might be missing from the initial clone. You must ensure `small_HateXplain_dataset.csv` and `precomputed_embeddings.npy` are present in the root directory, or the application will crash upon target identification.
4. Run the application using the terminal command: `streamlit run web.py`

**References**
*   Mathew, R. et al., "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"
*   Kim, J., Lee, B., and Sohn, K.-A., "Why Is It Hate Speech? Masked Rationale Prediction for Explainable Hate Speech Detection"

**Requirements**
*   Python 3.7+
*   PyTorch, Transformers (Hugging Face), LIME, Sentence-Transformers
*   Streamlit, Scikit-learn, Pandas, NumPy, FAISS

**Reproducing the Training Pipeline (Kaggle)**
If you wish to train the models from scratch rather than downloading the pre-trained weights:
1. Create a new Kaggle Notebook and upload `train_model.ipynb`.
2. Import the raw dataset `HateXplain_dataset.json` into your Kaggle environment and name the dataset folder `hatexplain-dataset`.
3. Run all cells in the notebook.
4. Upon completion, download the output weights from the Kaggle working directory:
   *   `bert_mlm_model.zip` (Optional: The Phase 1 MLM checkpoint)
   *   `bert_cf_model.zip` (Required: The Phase 2 Sequence Classification checkpoint used for inference)

**Data & Model Weights (External Links)** 
Due to GitHub file size limitations, the massive original training artifacts are hosted externally. **Note: You do not need to download these to run the web application**, as the necessary extracted weights and lightweight datasets are already included in this repository.
*   **Raw Dataset:** https://drive.google.com/file/d/1oeDZXB8jyCFbxTxm__T54MhO1NSdu3x9/edit *(Optional: Only required if you wish to run `extract_small_dataset.py` yourself to recreate the data processing step)*
*   **Phase 1 (MLM) Checkpoint:** https://drive.google.com/file/d/1QZ25-llvu0KDWflQkIfvxKDw61l95FZp/edit *(Optional: for inspecting intermediate training)*
*   **Phase 2 (Classification) Checkpoint:** https://drive.google.com/file/d/1vv719nIP8HsXzKIuBMJ3HGzb6YDLFfi7/edit *(Optional: The extracted model weights are already provided in the `final_fine_tuned_bert_2_class/` directory of this repo)*

**Running the Web Application Locally**
Because the final fine-tuned model directory (`final_fine_tuned_bert_2_class/`) and the preprocessed nearest-neighbor lookup table (`small_HateXplain_dataset.csv`) are already tracked in this repository, deploying the frontend is fast and straightforward.

1. **Clone the Repository:** Clone this project to your local machine and navigate into the root directory.
2. **Install Dependencies:** Ensure your Python environment has the necessary packages installed (e.g., `streamlit`, `torch`, `transformers`, `lime`, `sentence-transformers`, `faiss-cpu`, `pandas`, `scikit-learn`).
3. **Launch the App:** Run the following command in your terminal to start the Streamlit server:
   `streamlit run web_application.py`
4. The web application will automatically open in your default browser at `http://localhost:8501`.

**Optional: Recreating the Local Dataset**
If you wish to test the data engineering script and see how the nearest-neighbor lookup table was built:
1. Download `HateXplain_dataset.json` from the external links above and place it in the root directory.
2. Run `python extract_small_dataset.py` in your terminal. This will parse the JSON and regenerate `small_HateXplain_dataset.csv`.