# 2.1.1 Data Cleansing stage
# Manage Stemmer and Lemmatizer
import re
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing with stemming and lemmatization
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Split into words
    text = [stemmer.stem(lemmatizer.lemmatize(word)) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)  # Rejoin words back into text
    return text

# Apply pre-processing with stemming
bbc_data['cleaned_data'] = bbc_data['data'].apply(preprocess_text)

# VADER diagram process
from IPython import display
display.Image("VADER.png")

# Example of VADER calculation scores
{
  "pos": 0.33,       # 33% of the sentence is positive (from "good").
  "neg": 0.44,       # 44% of the sentence is negative (from "terrible").
  "neu": 0.23,       # 23% of the sentence is neutral (from other words like "the", "is").
  "compound": -0.34  # Overall sentiment: slightly negative.
}


