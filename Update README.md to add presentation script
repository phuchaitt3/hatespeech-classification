This Python code creates a web application using Streamlit that functions as a "Transformer-based NLP Hate Speech Classifier." Built on the HateXplain benchmark dataset, it allows users to input text and classify it into one of three categories: **Normal, Hate Speech, or Offensive**. Additionally, it provides an explanation for the prediction via LIME and identifies the potential target community of the hate speech.

Here's a detailed breakdown of the code's functionality:

**The Dataset (HateXplain) & Preprocessing**

The model is trained on **HateXplain**, a benchmark dataset for hate speech detection. The dataset covers multiple target communities: African, Islam, Jewish, Homosexual, Women, Refugee, Arab, Caucasian, Hispanic, and Asian.

To prepare the raw data for the model, the following preprocessing pipeline was implemented:
*   **Majority Voting Labeling:** Each post was annotated by multiple people. The final label (Normal, Hate Speech, Offensive) and the final Target Community were chosen based on a >50% majority consensus. Posts lacking a majority consensus were dropped.
*   **Rationale Extraction:** Annotators provided "rationales" (tokens they deemed critical for identifying hate speech). These were averaged; tokens scoring >0.5 were flagged as important (1), and the rest as (0), guiding the model's attention.
*   **Data Cleaning:** Unnecessary HTML tags were stripped, and empty targets were labeled as 'None'.
*   **Final Class Distribution:** The preprocessing resulted in a highly balanced dataset: Normal (7,814 samples), Hate Speech (5,935 samples), and Offensive (5,480 samples). The data was split into 80% for training and 20% for testing.

**Model Architecture & Training Pipeline**

The core classification model is based on BERT. To ensure the model deeply understood the linguistic nuances of hate speech before classification, a two-stage training approach was utilized:
1.  **Stage 1 - BERT Masked Language Model (MLM):** The pre-trained BERT model was first fine-tuned on the HateXplain corpus using a Masked Language Modeling objective. This allowed the model's embeddings to adapt to the specific vocabulary and slang of offensive internet text.
2.  **Stage 2 - Sequence Classification:** The MLM-adapted model was then passed to a `BertForSequenceClassification` head and fine-tuned on the 3-class labeled dataset to make the final predictions.

**Core Libraries and Their Roles:**

*   **Streamlit:** This library is used to create the web application's user interface. It allows for the creation of elements like text areas, buttons, sliders, and checkboxes with simple Python commands.
*   **PyTorch and Transformers (Hugging Face):** These are used for the machine learning aspect of the application.
    *   `BertForSequenceClassification` is a pre-trained deep learning model (BERT) that has been fine-tuned for text classification tasks. In this case, it is trained to classify text into 3 classes: "normal", "hate speech", and "offensive".
    *   `BertTokenizer` is used to prepare the input text so that it can be understood by the BERT model. This involves breaking the text into tokens (words or subwords) and converting them into numerical representations.
*   **LIME (Local Interpretable Model-agnostic Explanations):** LIME is a technique used to explain the predictions of machine learning models. It works by creating many small variations of the input text (by removing words) and seeing how the model's prediction changes. This helps to identify which words were most influential in the model's decision.
    *   `LimeTextExplainer` is the specific LIME tool used for text-based models.
*   **Sentence-Transformers:** This library provides an easy way to compute embeddings for sentences. Embeddings are numerical representations of text that capture its meaning.
    *   `paraphrase-MiniLM-L6-v2` is a specific pre-trained model that is efficient at creating these sentence embeddings.
*   **FAISS (Facebook AI Similarity Search):** This is a library developed by Facebook AI for efficient similarity searching of large datasets of embeddings. It allows for quickly finding the most similar items in a dataset to a given query item.
    *   `IndexFlatL2` is a specific type of FAISS index that performs a brute-force search using the L2 (Euclidean) distance to find the nearest neighbors.
*   **Pandas:** Used for data manipulation and reading the CSV file that contains the training data.
*   **Scikit-learn (`LabelEncoder`):** This is used to convert the text labels (e.g., "normal", "hate speech") into numerical values that the model can work with.
*   **NumPy:** A fundamental library for numerical operations in Python, used here for handling the embeddings.

**How the Application Works:**

1.  **Initialization and Caching:**
    *   The code starts by importing all the necessary libraries.
    *   Streamlit's `@st.cache_resource` decorator is used to cache the loading of the machine learning models and the creation of the FAISS index. This is a crucial optimization that prevents these time-consuming operations from being re-executed every time a user interacts with the app.

2.  **User Interface (Streamlit):**
    *   A title "Hate Speech Classifier" is displayed.
    *   A text area is provided for the user to enter text.
    *   A checkbox allows the user to enable or disable the LIME explanation.
    *   A slider lets the user choose how many words to show in the LIME explanation.
    *   Another checkbox allows the user to opt-in to finding the target of the hate speech.
    *   A "Classify" button triggers the main analysis process.
    *   The concept of `st.session_state` is used to maintain the state of the widgets (slider value, checkbox status) across user interactions, as Streamlit reruns the script on each interaction.

3.  **Core Logic (`explain_prediction` function):**
    *   **Input Validation:** It first checks if the user has entered any text.
    *   **LIME Explanation (Optional):**
        *   If the "Include LIME explanation" checkbox is ticked, it initializes the `LimeTextExplainer`.
        *   It defines a `model_predict` function that takes text, tokenizes it, and returns the model's prediction probabilities.
        *   The `explainer.explain_instance` method is then called. This is the core of LIME, which generates the explanation by perturbing the input text.
        *   The resulting explanation is then displayed as an HTML object using `st.components.v1.html`. This highlights the influential words in the original text.
    *   **Prediction:**
        *   The input text is tokenized and fed to the fine-tuned BERT model to get a prediction.
        *   The predicted label (e.g., "hatespeech" or "normal") is then displayed to the user.
    *   **Finding the Target of Hate Speech (Optional):**
        *   If the user has enabled this option and the prediction is not "normal," the code attempts to identify the target of the hate speech.
        *   It splits the input text into sentences and generates a sentence embedding for each using the `SentenceTransformer` model.
        *   These sentence embeddings are averaged to create a single representative embedding for the entire input text.
        *   This combined embedding is then used to search the pre-computed FAISS index. The index contains embeddings for all the text in the training dataset (`small_HateXplain_dataset.csv`).
        *   FAISS efficiently finds the most similar text from the dataset.
        *   The "target" group (e.g., African, Islam, Jewish, Homosexual, Women, Refugee, Arab, Caucasian, Hispanic, or Asian) associated with that most similar text in the original DataFrame is then displayed as the likely target of the hate speech.
        *   A confidence score is calculated based on the distance between the input text's embedding and the found similar text's embedding.

In essence, this code provides a sophisticated tool for not only identifying potential hate speech but also for understanding why the model made its decision (through LIME) and who the potential target might be (through similarity search with FAISS). The use of caching and a well-structured Streamlit interface makes it a user-friendly and efficient application.

**Running the Application Locally**

If you clone this repository to run locally, please note that two large dependency files are required in the root directory for the application to function:
1.  `small_HateXplain_dataset.csv`: The preprocessed text dataset used for nearest-neighbor target lookup.
2.  `precomputed_embeddings.npy`: The Sentence-Transformer embeddings required to build the FAISS index quickly without recalculating embeddings on boot.
