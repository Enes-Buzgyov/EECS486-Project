import pandas as pd
import re
import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import sys
import nltk


# Ensure required NLTK datasets are downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load stopwords once to improve efficiency
english_stopwords = set(stopwords.words("english"))


def clean_text(text):
    """Cleans text by lowering case, removing numbers, URLs, English stopwords."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in english_stopwords]
    return " ".join(tokens)


def is_english(text):
    """Determines if the given text is in English."""
    try:
        if pd.isna(text):
            return False  # Skip processing if text is NaN
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def clean_comments(artist_name):
    """Cleans comments for a given artist and writes cleaned data to output_csv."""
    input_csv = os.path.join("data", f"{artist_name}_comments.csv")
    output_csv = os.path.join("data", f"cleaned_{artist_name}_comments.csv")

    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv, header=None, names=["comment"], skip_blank_lines=True)

    # Filter out non-English comments
    df["is_english"] = df["comment"].apply(is_english)
    df = df[df["is_english"]]

    # Clean the comments
    df["cleaned_comment"] = df["comment"].apply(clean_text)

    # Apply comment character thresholds
    df["length"] = df["cleaned_comment"].str.len()
    df["length_category"] = pd.cut(
        df["length"],
        bins=[0, 20, 40, 150, float("inf")],
        labels=["too_short", "short", "mid_length", "long"],
    )

    # Remove comments that are too short
    df = df[df["length_category"] != "too_short"]

    # Save cleaned comments to CSV
    df[["cleaned_comment"]].to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)

    # Performance metrics
    total_comments = len(df.index)
    removed_comments = df["length_category"].value_counts().get("too_short", 0)
    short_comments = df["length_category"].value_counts().get("short", 0)
    mid_length_comments = df["length_category"].value_counts().get("mid_length", 0)
    long_comments = df["length_category"].value_counts().get("long", 0)

    # Print performance metrics
    print(f"Total comments: {total_comments}")
    print(f"Removed comments (too short): {removed_comments}")
    print(f"Short comments: {short_comments}")
    print(f"Mid-length comments: {mid_length_comments}")
    print(f"Long comments: {long_comments}")
    print(f"Cleaned comments have been saved to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_comments.py 'artist_name'")
        sys.exit(1)

    artist_name = sys.argv[1].replace(
        " ", "_"
    )  # Replace spaces with underscores for filename
    clean_comments(artist_name)
