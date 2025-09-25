from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(bbc_data['cleaned_data'], bbc_data['sentiment_label'], test_size=0.2, random_state=42)

# Display the sizes of the training and testing sets
print(f'Training set size: {len(X_train)}')
print(f'Testing set size: {len(X_test)}')

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate Naive Bayes: 0 = negative, 1 = neutral, 2 = positive
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Generate classification report
report = classification_report(y_test, y_pred_nb, output_dict=True)

print("Naive Bayes Classifier:")

# Data grouping
grouped_metrics = pd.DataFrame(report).transpose()
print(grouped_metrics)

# Visualize precision, recall, and F1-score for each sentiment
grouped_metrics[['precision', 'recall', 'f1-score']].iloc[:-3].plot(kind='bar')

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate SVM: 0 = negative, 1 = neutral, 2 = positive
y_pred_svm = svm_classifier.predict(X_test_tfidf)

print("Support Vector Machine Classifier:")

# Data grouping
grouped_metrics = pd.DataFrame(report).transpose()
print(grouped_metrics)

# Visualize precision, recall, and F1-score for each sentiment
grouped_metrics[['precision', 'recall', 'f1-score']].iloc[:-3].plot(kind='bar')

# Train Logistic Regression classifier
log_reg_classifier = LogisticRegression(max_iter=1000)
log_reg_classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate Logistic Regression: 0 = negative, 1 = neutral, 2 = positive
y_pred_log_reg = log_reg_classifier.predict(X_test_tfidf)

print("Logistic Regression Classifier:")

# Data grouping
grouped_metrics = pd.DataFrame(report).transpose()
print(grouped_metrics)

# Visualize precision, recall, and F1-score for each sentiment
grouped_metrics[['precision', 'recall', 'f1-score']].iloc[:-3].plot(kind='bar')
