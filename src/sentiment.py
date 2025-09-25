# Import Sentiment Analyser by creating the 'analyser' as a variable for storing for the later text after classifying into integer score  
nltk.download('vader_lexicon')
analyser = SentimentIntensityAnalyzer()

# Analyse sentiment for each review
bbc_data['sentiment'] = bbc_data['cleaned_data'].apply(lambda x: analyser.polarity_scores(x))

# Extract compound sentiment scores
bbc_data['compound_sentiment'] = bbc_data['sentiment'].apply(lambda x: x['compound'])

# Display the first few rows with sentiment scores
print(bbc_data[['data', 'cleaned_data', 'compound_sentiment']].head())

# Encoding data
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(bbc_data['cleaned_data'])

# Data Modeling with NMF
n_topics = 5  # Number of topics
nmf = NMF(n_components=n_topics, random_state=42)
nmf.fit(X)

# Print topics
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    print(f"\nTopic #{topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialise Sentiment Analyser
analyser = SentimentIntensityAnalyzer()

# Analyse sentiment for each description
bbc_data['sentiment_scores'] = bbc_data['cleaned_data'].apply(lambda x: analyser.polarity_scores(x))

# Extract positive, negative, and neutral sentiment scores
bbc_data['positive_score'] = bbc_data['sentiment_scores'].apply(lambda x: x['pos'])
bbc_data['negative_score'] = bbc_data['sentiment_scores'].apply(lambda x: x['neg'])
bbc_data['neutral_score'] = bbc_data['sentiment_scores'].apply(lambda x: x['neu'])
bbc_data['compound_score'] = bbc_data['sentiment_scores'].apply(lambda x: x['compound'])

# Categorise the sentiment based on the compound score
bbc_data['sentiment'] = bbc_data['compound_score'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

# Display the first few rows with sentiment scores
print(bbc_data[['data', 'positive_score', 'negative_score', 'neutral_score', 'compound_score', 'sentiment']].head())

# Encode the sentiment labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
bbc_data['sentiment_label'] = label_encoder.fit_transform(bbc_data['sentiment'])

# Display the first few rows with sentiment labels
print(bbc_data[['data', 'cleaned_data', 'sentiment', 'sentiment_label']].head())
