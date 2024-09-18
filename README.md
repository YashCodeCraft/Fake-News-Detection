# Fake News Detection (NLP)

This project focuses on detecting fake news using Natural Language Processing (NLP) techniques. The goal is to classify news articles into "fake" or "real" categories based on their textual content. This README provides a comprehensive overview of the project, including dataset details, instructions for running the code, and other relevant information.

## Project Description

The Fake News Detection project utilizes various NLP techniques to preprocess and analyze news articles. The process involves:
1. **Loading and cleaning the dataset**.
2. **Preprocessing text data** with lemmatization and stopword removal.
3. **Feature extraction** using Count Vectorization with n-grams.
4. **Training a Naive Bayes classifier** to distinguish between fake and real news.
5. **Evaluating the model** with performance metrics and visualizing results with a confusion matrix.

## Dataset

- **File**: `dataset in csv format`
- **Description**: The dataset contains news articles with labels indicating whether each article is fake or real.
- **Columns**:
  - `id`: Unique identifier for each article
  - `title`: Title of the article
  - `author`: Author of the article
  - `text`: Full text of the article
  - `label`: The label indicating whether the news is fake or real

**Note**: The code processes a subset of the dataset (`df = df[:50]`) for demonstration purposes. Replace `"dataset in csv format"` with the actual file path of your dataset.

## Instructions

### Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install pandas nltk scikit-learn seaborn matplotlib
```
## Running the Code
1. Prepare the Dataset: Save your dataset in CSV format to your working directory. Update the file path in the main.py script as needed.

2. Run the Script: Execute the main.py script to process the dataset, train the model, and visualize the results.

```bash
python main.py
```

## Script Details
- `main.py:` Contains the implementation for fake news detection. The script includes:
    - Data loading and preprocessing.
    - Text cleaning and feature extraction using Count Vectorization.
    - Training of a Naive Bayes classifier.
    - Evaluation using classification metrics and confusion matrix visualization.
 
## Example Output
Upon running the script, you will see:
- Classification Report: Metrics such as precision, recall, F1-score, and accuracy.
- Confusion Matrix: A heatmap showing the performance of the classifier.

## Results
The script prints out the classification report and confusion matrix. The confusion matrix is visualized using Seaborn to provide a clear understanding of model performance.

## Visualizations
The confusion matrix is plotted using Seaborn:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cmap='YlGnBu')
plt.show()
```
