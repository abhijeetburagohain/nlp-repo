import spacy
import pandas as pd
import string
from tqdm import tqdm
import ast
import re
import emoji
from spellchecker import SpellChecker
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

import numpy as np
from sklearn.metrics import f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import matplotlib.pyplot as plt

import textstat
import spacy
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer

## Scaling the feautures 
from sklearn.preprocessing import StandardScaler


## Initial functions
## Loading the spacy 
nlp = spacy.load("en_core_web_sm")

def safe_lower(tokens):
    if isinstance(tokens, list):
        return [str(token).lower().strip() for token in tokens]
    return tokens

## Removing the stop words
stopwords = nlp.Defaults.stop_words
punctuations = string.punctuation

def clean_tokens(token_list):
    return [
        token for token in token_list
        if token.lower() not in stopwords and token not in punctuations
    ]

### FUNCTIONS FOR CLEANING THE UNWANTED SYMBOLS
def remove_prefix_noise(token):
    # Remove patterns like '1)', '2)', etc.
    token = re.sub(r'^\d+\)', '', token)

    # Remove patterns like '1.', '2.', etc. ONLY if not followed by a digit (e.g. 1.design, not 1.5)
    token = re.sub(r'^\d+\.(?=[a-zA-Z])', '', token)

    # Remove patterns like '3-', '4-', etc. at the start
    token = re.sub(r'^\d+-:-;-', '', token)

    return token


def remove_unwanted_symbols(token_list):
    bad_tokens = {"", "\n", "\\n", "..", "...", ".", "️", '....','\n\n','\n\n\n','.......','......',
                    '):','-)',"--------","----","---","--","-",'.....', '........', '.........', '..........', '............', '.............', '..............', '................',
                    '/-','-','-)',',',"#","&"}  # include the invisible '️'
    cleaned = [token.strip() for token in token_list if token.strip() not in bad_tokens]
    cleaned = [
        "" if token.lstrip("-/\\.#&") in bad_tokens else remove_prefix_noise(token.lstrip("-/\\.#&"))
        for token in cleaned
        ]
    return cleaned

# Normalize number strings inside individual tokens
def normalize_token(token):
    if isinstance(token, str):
        # Convert "23,000" → "23000"
        token = re.sub(r'(\d{1,3}(?:,\d{3})+)', lambda m: m.group(1).replace(',', ''), token)
        # Convert "23k"/"23K" → "23000"
        token = re.sub(r'^(\d+)[kKk.]$', lambda m: str(int(m.group(1)) * 1000), token)
        token = re.sub(r'^(\d+)(lakh|lakhs)$', lambda m: str(int(m.group(1)) * 100000), token, flags=re.IGNORECASE)
    return token

# Apply it to each token in the list
def normalize_token_list(token_list):
    return [normalize_token(token) for token in token_list]

hindi_to_eng = {
    "bakwas": "nonsense",
    "zabardast": "awesome",
    "achha": "good",
    "ghatiya": "bad",
    "mast": "great"
}

def translate_hindi_words(tokens):
    return [hindi_to_eng.get(word.lower(), word) for word in tokens]

### EMOJIS CLEANING [CONVERTING THEM TO TEXTS]
def demojize_tokens(tokens):
    return [emoji.demojize(token, language='en') for token in tokens]

spell = SpellChecker()

def count_spelling_errors(text):
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled)

def count_entities(text, label):
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ == label)

def preprocess(data,clean_tokens,safe_lower,remove_unwanted_symbols,normalize_token_list,
    translate_hindi_words,demojize_tokens):
    data['tokens'] = data['review'].apply(lambda x: [token.text for token in nlp(x)] if pd.notnull(x) else [])
    data['tokens'] = data['tokens'].apply(safe_lower)
    data['clean_tokens'] = data['tokens'].apply(clean_tokens)
    data['clean_tokens'] = data['clean_tokens'].apply(remove_unwanted_symbols)
    data['clean_tokens'] = data['clean_tokens'].apply(normalize_token_list)
    data['clean_tokens'] = data['clean_tokens'].apply(translate_hindi_words)
    data['lemmatized_tokens'] = data['clean_tokens'].apply(lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))])
    data['clean_tokens_demojize'] = data['lemmatized_tokens'].apply(demojize_tokens)
    
    return data

def feature_engineered(data,count_entities,count_spelling_errors):
    # Total number of characters across all clean tokens
    data['char_count'] = data['lemmatized_tokens'].apply(lambda x: sum(len(word) for word in x))
    data['word_count'] = data['lemmatized_tokens'].apply(lambda x: len(x))
    data['avg_word_length'] = data['lemmatized_tokens'].apply(lambda x: (sum(len(word) for word in x) / len(x)) if len(x) > 0 else 0)
    
    # applying textblob on review column
    data['sentiment_polarity'] = data['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['sentiment_subjectivity'] = data['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    ## duplicating the above polarity [fallback measure] [temporary]
    data['sentiment_polarity_2'] = data['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['sentiment_subjectivity_2'] = data['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    ## applying VADER
    analyzer = SentimentIntensityAnalyzer()
    data['vader_title'] = data['review'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    data['vader_review'] = data['review'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    data['vader_weighted_avg'] = data['review'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    ## getting the fleasch columns
    data['flesch_reading_ease'] = data['review'].apply(textstat.flesch_reading_ease)
    data['flesch_kincaid_grade'] = data['review'].apply(textstat.flesch_kincaid_grade)  

    ## getting the pos_tagging columns
    data['noun_count'] = data['review'].apply(lambda x: sum(1 for token in nlp(x) if token.pos_ == 'NOUN'))
    data['verb_count'] = data['review'].apply(lambda x: sum(1 for token in nlp(x) if token.pos_ == 'VERB'))
    data['adj_count'] = data['review'].apply(lambda x: sum(1 for token in nlp(x) if token.pos_ == 'ADJ'))

    ##getting the count of entities 
    data['person_count'] = data['review'].apply(lambda x: count_entities(x, 'PERSON'))
    data['org_count'] = data['review'].apply(lambda x: count_entities(x, 'ORG'))
    data['gpe_count'] = data['review'].apply(lambda x: count_entities(x, 'GPE'))

    ## CREATING THE TOPICS USING N_COMPONENTS==2
    # Convert token lists to space-joined strings
    data['lemmatized_text'] = data['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
    # Step 1: Vectorize the text
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(data['lemmatized_text'])
    # Step 2: Fit LDA model
    lda = LatentDirichletAllocation(n_components=2, random_state=42)
    lda.fit(doc_term_matrix)

    # Step 3: Get topic distributions for each document
    topic_distributions = lda.fit_transform(doc_term_matrix)

    # Step 4: Add each topic's weight as a feature
    for i in range(topic_distributions.shape[1]):
        data[f'topic_{i}'] = topic_distributions[:, i]
    
    ## getting the count of spelling error column
    data['count_of_spelling_errors'] = data['review'].apply(count_spelling_errors)
    return data

def get_input(data,tfidf,scaler):
    ## Selecting the featurs that are needed to incorpoarte into the models

    features=['sentiment_polarity', 'sentiment_subjectivity','sentiment_polarity_2', 'sentiment_subjectivity_2', 
                'vader_title','vader_review','vader_weighted_avg', 'flesch_reading_ease',
                'flesch_kincaid_grade','noun_count', 'verb_count', 'adj_count', 'person_count', 'org_count',
                'gpe_count','topic_0', 'topic_1','count_of_spelling_errors']

    manual_features = data[features]
    # scaler = StandardScaler()
    scaled_manual = scaler.transform(manual_features)

    data['clean_text'] = data['clean_tokens_demojize'].apply(lambda x: ' '.join(x))
    # tfidf = TfidfVectorizer(max_features=5000)  # You can adjust max_features
    X = tfidf.transform(data['clean_text'])
    print(X.shape)
    X = hstack([X, scaled_manual])

    return X,data