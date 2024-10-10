# News Article Classification

This project implements a machine learning pipeline to classify news articles into one of several predefined categories based on their titles and headlines. It uses `Logistic Regression` model, selected through hyperparameter tuning, achieving the best performance on the dataset.

## Dataset

- **Training Data**: `news_train.csv`
- **Test Data**: `test.csv`

Each dataset contains the following features:

- `News_title`: The title of the news article.
- `News_headline`: The headline of the news article.
- `Category`: The category label for training data.

## Preprocessing

The text data is preprocessed using the following steps:

1. **Text Lowercasing**: All characters are converted to lowercase.
2. **Punctuation Removal**: Punctuation and special characters are removed.
3. **Tokenization**: The text is split into individual words (tokens).
4. **Stopword Removal**: Common English stopwords are removed.
5. **Lemmatization**: Words are reduced to their base forms using the WordNet lemmatizer.
6. **NaN Values**: All NaN values are replaced by empty strings.

The `train_df['text']` column is created by concatenating the news title and headline.

## Feature Extraction

To convert the preprocessed text data into numerical features:

- **TF-IDF Vectorization** is applied to the `text` column.
  - Max features: 5000
  - N-gram range: (1, 2) (unigrams and bigrams)
  - Stopwords: English stop words are removed.(Not necessary as it has already been done in the preprocessing part.)

## Models

Four machine learning models are used in this project:

1. **Logistic Regression**
2. **Naive Bayes**
3. **Support Vector Machine (SVM)**
4. **Random Forest**

### Hyperparameter Tuning

Grid search with cross-validation is used to find the best hyperparameters for each model:

- **Logistic Regression**: Tuning `C` (inverse regularization strength) and `solver`.
- **Naive Bayes**: Tuning the `alpha` parameter.
- **SVM**: Tuning `C` and the kernel type (`linear`, `rbf`).
- **Random Forest**: Tuning the number of estimators and maximum tree depth.

### Best Model Selection

After performing grid search, **Logistic Regression (C=10, solver='liblinear')** was chosen as the best model based on validation accuracy.

## Training and Validation

The Logistic Regression model is trained on the entire training dataset, and the performance is validated on the validation dataset using:

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)

## Evaluation Metrics

- **Validation Accuracy**: Reported for the validation dataset.
- **Precision, Recall, F1-Score**: Calculated for each category to evaluate model performance.

## Category Mapping

The news categories are mapped as follows:

- `Arts` → 0
- `Business` → 1
- `Humour` → 2
- `Politics` → 3
- `Sports` → 4
- `Tech` → 5

## Submission

The predictions for the test dataset are made using the best model (Logistic Regression) and stored in a CSV file (`submission.csv`). The format of the submission file is as follows:

- `ID`: The ID of the test sample.
- `Category`: The predicted category for each test sample.

## Results

- **Validation Accuracy**: The best validation accuracy achieved is ~85%.
- The test predictions are giving a weighted F1 score of ~0.83 in Kaggle competition.
