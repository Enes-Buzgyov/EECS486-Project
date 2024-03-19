import pandas as pd
import os
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat.textstat import textstatistics
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

nltk.download("vader_lexicon")


# Function to calculate type-token ratio (TTR)
def type_token_ratio(text):
    tokens = word_tokenize(text)
    return len(set(tokens)) / len(tokens)


# Function to perform sentiment analysis
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


# Function to perform readability test
def readability_test(text):
    return textstatistics().flesch_reading_ease(text)


# Function to generate word cloud
def generate_word_cloud(fdist):
    wordcloud = WordCloud(width=800, height=400, background_color="white")
    wordcloud.generate_from_frequencies(fdist)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def main(input_directory):
    # Read each CSV and perform analysis
    for file in os.listdir(input_directory):
        if file.endswith("_comments.csv"):
            file_path = os.path.join(input_directory, file)
            df = pd.read_csv(file_path)
            all_text = " ".join(df["cleaned_comment"].dropna().tolist())

            # Tokenization
            tokens = word_tokenize(all_text)
            # Frequency Distribution
            fdist = FreqDist(tokens)
            # Vocabulary Richness (TTR)
            ttr = type_token_ratio(all_text)
            # Lexical Diversity
            lexical_diversity = fdist.hapaxes()
            # Sentiment Analysis
            sentiment = sentiment_analysis(all_text)
            # Readability Scores
            readability = readability_test(all_text)
            # Collocations
            bigram_measures = BigramAssocMeasures()
            finder = BigramCollocationFinder.from_words(tokens)
            collocations = finder.nbest(bigram_measures.pmi, 10)
            # Word Cloud
            generate_word_cloud(fdist)

            # Further analysis like POS Tagging, Concordance, NER, etc. can be added as needed

            # Output analysis results
            print(f"Analysis for {file}:")
            print(f"Top 10 Words: {fdist.most_common(10)}")
            print(f"Type-Token Ratio: {ttr}")
            print(f"Unique Words: {len(lexical_diversity)}")
            print(f"Sentiment Scores: {sentiment}")
            print(f"Readability Score: {readability}")
            print(f"Top 10 Collocations: {collocations}")
            print("\n\n")


if __name__ == "__main__":
    # Adjust the directory path to where you have your cleaned CSV files
    input_directory = "output"
    main(input_directory)
