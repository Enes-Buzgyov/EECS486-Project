# Set of funciotns that help us get a basic understanding of comments. Does tf idf, bigrams, etc.
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import (
    BigramAssocMeasures,
    BigramCollocationFinder,
    TrigramAssocMeasures,
    TrigramCollocationFinder,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob


def preprocess_comments(comments):
    preprocessed_comments = []
    lemmatizer = WordNetLemmatizer()

    for comment in comments:
        # Tokenize the comment into sentences
        sentences = sent_tokenize(comment.strip())
        preprocessed_sentences = []

        for sentence in sentences:
            # Tokenize the sentence into words
            tokens = word_tokenize(sentence)
            # Remove stop words and punctuation, and lemmatize the words
            tokens = [
                lemmatizer.lemmatize(token.lower())
                for token in tokens
                if token.isalnum() and token.lower() not in stopwords.words("english")
            ]
            preprocessed_sentences.append(" ".join(tokens))

        preprocessed_comments.append(" ".join(preprocessed_sentences))

    return preprocessed_comments


def create_baseline_model(artist_name):
    cleaned_comments_file = os.path.join("data", f"cleaned_{artist_name}_comments.csv")

    if not os.path.exists(cleaned_comments_file):
        print(f"Cleaned comments file not found for artist: {artist_name}")
        return None

    comments = []

    with open(cleaned_comments_file, "r", encoding="utf-8") as file:
        comments = file.readlines()

    preprocessed_comments = preprocess_comments(comments)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_comments)

    # Get the vocabulary and IDF values
    vocabulary = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_

    # Perform topic modeling using Latent Dirichlet Allocation (LDA)
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(tfidf_matrix)

    # Extract bigrams and trigrams
    bigram_finder = BigramCollocationFinder.from_documents(
        [doc.split() for doc in preprocessed_comments]
    )
    trigram_finder = TrigramCollocationFinder.from_documents(
        [doc.split() for doc in preprocessed_comments]
    )

    # Calculate bigram and trigram association measures
    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()

    # Create a dictionary to store the baseline model
    baseline_model = {
        "vocabulary": vocabulary,
        "idf_values": idf_values,
        "lda_model": lda_model,
        "bigram_finder": bigram_finder,
        "trigram_finder": trigram_finder,
        "bigram_measures": bigram_measures,
        "trigram_measures": trigram_measures,
    }

    return baseline_model


def analyze_baseline_model(baseline_model):
    vocabulary = baseline_model["vocabulary"]
    idf_values = baseline_model["idf_values"]
    lda_model = baseline_model["lda_model"]
    bigram_finder = baseline_model["bigram_finder"]
    trigram_finder = baseline_model["trigram_finder"]
    bigram_measures = baseline_model["bigram_measures"]
    trigram_measures = baseline_model["trigram_measures"]

    # Print the top 10 words with the highest IDF values
    top_words = sorted(zip(vocabulary, idf_values), key=lambda x: x[1], reverse=True)[
        :10
    ]
    print("Top 10 words with highest IDF values:")
    for word, idf in top_words:
        print(f"{word}: {idf}")

    # Print the top 10 bigrams with the highest PMI scores
    top_bigrams = bigram_finder.nbest(bigram_measures.pmi, 10)
    print("\nTop 10 bigrams with highest PMI scores:")
    for bigram in top_bigrams:
        print(" ".join(bigram))

    # Print the top 10 trigrams with the highest PMI scores
    top_trigrams = trigram_finder.nbest(trigram_measures.pmi, 10)
    print("\nTop 10 trigrams with highest PMI scores:")
    for trigram in top_trigrams:
        print(" ".join(trigram))

    # Print the topic-word distributions from LDA
    print("\nTopic-Word Distributions:")
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{topic_idx+1}:")
        print(", ".join([vocabulary[i] for i in topic.argsort()[: -10 - 1 : -1]]))


def main():
    artist_name = "Pink Floyd"
    baseline_model = create_baseline_model(artist_name)
    if baseline_model:
        print(f"Baseline Model for Artist: {artist_name}\n")
        analyze_baseline_model(baseline_model)
    else:
        print("Failed to create the baseline model.")


if __name__ == "__main__":
    main()
