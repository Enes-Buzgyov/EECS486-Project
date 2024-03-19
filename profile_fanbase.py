import pandas as pd
import spacy
import os
import gensim
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure spaCy's English model is loaded (run "python -m spacy download en_core_web_sm" if necessary)
nlp = spacy.load("en_core_web_sm")


def sentiment_analysis(comments):
    sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
    average_sentiment = sum(sentiments) / len(sentiments)
    return average_sentiment


def frequent_terms(comments, n=10):
    vectorizer = CountVectorizer(stop_words="english", max_features=n)
    term_matrix = vectorizer.fit_transform(comments)
    terms = vectorizer.get_feature_names_out()
    frequencies = term_matrix.sum(axis=0).A1
    return sorted(zip(terms, frequencies), key=lambda x: x[1], reverse=True)


def extract_entities(comments):
    entities = [ent.text for comment in comments for ent in nlp(comment).ents]
    return Counter(entities).most_common(10)


def perform_lda(comments):
    processed_comments = [
        gensim.utils.simple_preprocess(comment) for comment in comments
    ]
    dictionary = corpora.Dictionary(processed_comments)
    corpus = [dictionary.doc2bow(text) for text in processed_comments]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)
    return topics


def generate_feature_vectors(comments):
    avg_sentiment = sentiment_analysis(comments)
    terms_freq = dict(frequent_terms(comments))
    entities_freq = dict(extract_entities(comments))

    lda_features = {}
    for topic_info in perform_lda(comments):
        # The topic_info format is (topic_number, "prob*word + prob*word + ...")
        _, topic_terms = topic_info
        for term_prob_pair in topic_terms.split(" + "):
            prob, word = term_prob_pair.split("*")
            word = word.strip().replace('"', "")
            prob = float(prob.strip())
            lda_features[word] = prob

    features = {
        **terms_freq,
        **entities_freq,
        **lda_features,
        "avg_sentiment": avg_sentiment,
    }
    return features


def compare_fan_bases(fanbase_profiles):
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(pd.DataFrame(fanbase_profiles).fillna(0))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feature_matrix)
    tsne = TSNE(n_components=2, perplexity=50, n_iter=300)
    tsne_result = tsne.fit_transform(feature_matrix)
    similarity_matrix = cosine_similarity(feature_matrix)
    return pca_result, tsne_result, similarity_matrix


def main(input_directory):
    fanbase_profiles = {}

    for file in os.listdir(input_directory):
        if file.endswith("_comments.csv"):
            file_path = os.path.join(input_directory, file)
            df = pd.read_csv(file_path)
            comments = df["cleaned_comment"].dropna().tolist()
            fanbase_profiles[file] = generate_feature_vectors(comments)

    pca_result, tsne_result, similarity_matrix = compare_fan_bases(fanbase_profiles)

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
    plt.title("PCA of Fan Base Features")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1])
    plt.title("t-SNE of Fan Base Features")
    plt.show()

    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm")
    plt.title("Fan Base Similarity")
    plt.xticks(
        ticks=range(len(fanbase_profiles)), labels=fanbase_profiles.keys(), rotation=90
    )
    plt.yticks(
        ticks=range(len(fanbase_profiles)), labels=fanbase_profiles.keys(), rotation=0
    )
    plt.show()


if __name__ == "__main__":
    input_directory = "output"  # or whatever directory you have
    main(input_directory)
