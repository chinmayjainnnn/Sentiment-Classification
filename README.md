---

# Sentiment Analysis Project

This project demonstrates the basics of text processing by building text classifiers and learning to represent words effectively. It includes techniques such as rule-based classification, Bag-of-Words (BoW), and Word2Vec embeddings.

## Dataset

The dataset consists of sentences labeled with two sentiments: `positive` and `negative`. Examples include:

- **Positive**: *"I really like your new haircut!"*
- **Negative**: *"Your new haircut is awful!"*

The dataset is divided into three parts:
1. **Training Set**: `train_data.csv` (provided)
2. **Validation Set**: `val_data.csv` (provided)
3. **Test Set**: `test_data.csv` (not provided for blind evaluation)

### Downloading the Dataset

To download the dataset, execute the following commands:

```bash
# Training data
wget -O train_data.csv "https://docs.google.com/spreadsheets/d/176-KrOP8nhLpoW91UnrOY9oq_-I0XYNKS1zmqIErFsA/gviz/tq?tqx=out:csv&sheet=train_data.csv"

# Validation data
wget -O val_data.csv "https://docs.google.com/spreadsheets/d/1YxjoAbatow3F5lbPEODToa8-YWvJoTY0aABS9zaXk-c/gviz/tq?tqx=out:csv&sheet=val_data.csv"

# Test data
wget -O test_data.csv "https://docs.google.com/spreadsheets/d/1YxjoAbatow3F5lbPEODToa8-YWvJoTY0aABS9zaXk-c/gviz/tq?tqx=out:csv&sheet=test_data.csv"
```

## Python Libraries Required

Ensure the following Python libraries are installed:

- **General libraries**: `numpy`, `pandas`, `re`
- **Machine Learning**: `sklearn`
- **Visualization**: `matplotlib`, `seaborn`
- **Word2Vec and NLP**: `gensim`, `torch`, `torchtext`
- **Utility**: `wget`, `tqdm`

Install all required libraries via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn gensim torch torchtext tqdm wget
```
- you will have to deal with the dependency issues - ALL THE BEST!!!

## Methodology

### Part I: Rule-Based Sentiment Classification
1. **Feature Extraction**: Handcrafted rules identify keywords indicative of positive or negative sentiment.
2. **Prediction**: A linear scoring model based on the extracted features predicts sentiment.
3. **Evaluation**: The rule-based model achieves accuracy ~60%, showcasing the importance of keyword-based analysis.

### Part II: Bag-of-Words (BoW)
1. **Vectorization**: The text is converted into numerical vectors using word frequency.
2. **Learning Weights**: Logistic regression is used to learn feature weights for classification.
3. **Evaluation**: Accuracy improves with automated weight learning compared to manual assignment.

### Part III: Word Embeddings using Word2Vec
1. **Word2Vec Training**: Skip-gram model implemented from scratch using PyTorch to learn word embeddings.
2. **Visualization**: PCA reduces embeddings to 2D for visualization of semantic similarity.
3. **Analogy Tests**: Evaluate embedding quality on analogy questions like *"man:king :: woman:?"*.

### Evaluation Metrics
- **Classification Accuracy**: Measures correctness of predictions.
- **Word Similarity**: Assesses semantic proximity between words using cosine similarity.
- **Analogy Precision**: Tests embedding performance on analogy tasks, comparing the top 5 predictions.

## How to Run


1. **Download Dataset**:
    Follow the dataset download instructions above.

2. **Run the file to explore:**:
    - Rule-based models
    - Bag-of-Words classification
    - Word embeddings via Word2Vec

4. **Evaluate Results**:
    Review accuracy metrics, word similarity plots, and analogy test outcomes.

---