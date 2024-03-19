import pandas as pd
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import nltk
import os

# Ensure required NLTK datasets are downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load stopwords once to improve efficiency
english_stopwords = set(stopwords.words("english"))


def clean_text(text):
    """
    Cleans text by lowering case, removing special characters, numbers, URLs,
    stopwords, and applying lemmatization.
    """
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in english_stopwords]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def is_english(text):
    """
    Determines if the given text is English. Handles cases where text might
    not be a string and avoids language detection on empty strings.
    """
    try:
        text = str(text).strip()
        if not text:
            return False
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_comments(input_csv, output_csv):
    """
    Reads comments from input_csv, filters non-English comments,
    cleans the remaining comments, and writes them to output_csv.
    """
    # Check if input CSV exists to avoid FileNotFoundError
    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv, names=["comment"], header=None, skiprows=1)

    # Filter out non-English comments
    df["is_english"] = df["comment"].apply(is_english)
    df = df[df["is_english"]]

    # Clean the filtered English comments
    df["cleaned_comment"] = df["comment"].apply(clean_text)
    df[["cleaned_comment"]].to_csv(
        output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"English comments have been cleaned and saved to {output_csv}")
