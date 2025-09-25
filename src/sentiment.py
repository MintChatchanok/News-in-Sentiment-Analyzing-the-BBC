# Import Sentiment Analyser by creating the 'analyser' as a variable for storing for the later text after classifying into integer score  
nltk.download('vader_lexicon')
analyser = SentimentIntensityAnalyzer()

# Analyse sentiment for each review
bbc_data['sentiment'] = bbc_data['cleaned_data'].apply(lambda x: analyser.polarity_scores(x))

# Extract compound sentiment scores
bbc_data['compound_sentiment'] = bbc_data['sentiment'].apply(lambda x: x['compound'])

# Display the first few rows with sentiment scores
print(bbc_data[['data', 'cleaned_data', 'compound_sentiment']].head())
