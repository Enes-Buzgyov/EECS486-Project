import pandas as pd
import re
import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import sys
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from collections import Counter
import string

# Ensure required NLTK datasets are downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load stopwords once to improve efficiency
english_stopwords = set(stopwords.words("english"))


def clean_text(text):
    """Cleans text by lowering case, removing numbers, URLs, punctuation, and English stopwords."""
    text = text.lower()
    text = re.sub(r"\\d+", "", text)  # Remove numbers
    text = re.sub(r"http\\S+", "", text)  # Remove URLs
    text = re.sub(
        r"\[^a-zA-Z\\s\]", "", text
    )  # Remove special characters and punctuation
    text = re.sub(r"(.)\\1+", "\\1", text)  # Remove repeated characters
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens if word not in english_stopwords and len(word) > 1
    ]
    return " ".join(tokens)


def is_english(text):
    """Determines if the given text is in English."""
    try:
        if pd.isna(text):
            return False  # Skip processing if text is NaN
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def get_top_bigrams(comments, n=10):
    """Finds the top n bigrams in the comments."""
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(
        [word_tokenize(comment) for comment in comments]
    )
    finder.apply_freq_filter(3)  # Filter out bigrams that occur less than 3 times
    top_bigrams = finder.nbest(bigram_measures.pmi, n)
    return top_bigrams


def get_top_trigrams(comments, n=10):
    """Finds the top n trigrams in the comments."""
    trigram_measures = TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_documents(
        [word_tokenize(comment) for comment in comments]
    )
    finder.apply_freq_filter(3)  # Filter out trigrams that occur less than 3 times
    top_trigrams = finder.nbest(trigram_measures.pmi, n)
    return top_trigrams


def get_word_frequency(comments, n=10):
    """Finds the top n most frequent words in the comments."""
    words = [
        word
        for comment in comments
        for word in word_tokenize(comment)
        if word not in string.punctuation
    ]
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)
    return top_words


def clean_comments(artist_name):
    """Cleans comments for a given artist and writes cleaned data to output_csv."""
    artist_name_normalized = artist_name.replace(" ", "_")
    input_csv = os.path.join("data", f"{artist_name_normalized}_comments.csv")
    output_csv = os.path.join("data", f"cleaned_{artist_name_normalized}_comments.csv")
    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv, header=None, names=["comment"], skip_blank_lines=True)
    total_comments = len(df.index)

    # Filter out non-English comments
    df["is_english"] = df["comment"].apply(is_english)
    df = df[df["is_english"]]

    # Clean the comments
    df["cleaned_comment"] = df["comment"].apply(clean_text)

    # Apply comment character thresholds
    df["length"] = df["cleaned_comment"].str.len()
    df["length_category"] = pd.cut(
        df["length"],
        bins=[0, 3, 40, 150, float("inf")],
        labels=["too_short", "short", "mid_length", "long"],
    )

    # Remove comments that are too short
    df = df[df["length_category"] != "too_short"]

    # Save cleaned comments to CSV
    df[["cleaned_comment"]].to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)

    # Get top bigrams, trigrams, and word frequency
    comments = df["cleaned_comment"].tolist()
    top_bigrams = get_top_bigrams(comments)
    top_trigrams = get_top_trigrams(comments)
    top_words = get_word_frequency(comments)

    # Performance metrics
    cleaned_comments = len(df.index)
    removed_comments = total_comments - cleaned_comments
    short_comments = len(df[df["length_category"] == "short"])
    mid_length_comments = len(df[df["length_category"] == "mid_length"])
    long_comments = len(df[df["length_category"] == "long"])

    # Print performance metrics and insights
    print(f"Total comments: {total_comments}")
    print(f"Cleaned comments: {cleaned_comments}")
    print(f"Removed comments (non-English or too short): {removed_comments}")
    print(f"Short comments: {short_comments}")
    print(f"Mid-length comments: {mid_length_comments}")
    print(f"Long comments: {long_comments}")
    print(f"Top 10 bigrams: {top_bigrams}")
    print(f"Top 10 trigrams: {top_trigrams}")
    print(f"Top 10 most frequent words: {top_words}")
    print(f"Cleaned comments have been saved to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_comments.py 'artist_name'")
        sys.exit(1)

    artist_name = sys.argv[1].replace(
        " ", "_"
    )  # Replace spaces with underscores for filename
    clean_comments(artist_name)
