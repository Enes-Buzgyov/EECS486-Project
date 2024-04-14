import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from tqdm import tqdm
import sys
from fetch_comments import normalize_artist_name


def perform_comment_analysis(artist_name):
    # Configure logging
    logging.basicConfig(filename="comment_analysis.log", level=logging.INFO)

    normalized_artist_name = normalize_artist_name(artist_name)
    csv_file = f"cleaned_{normalized_artist_name}_comments.csv"

    print(f"Analyzing comments for artist: {artist_name}")
    logging.info(f"Analyzing comments for artist: {artist_name}")

    data_folder = "data"
    comments_df = pd.read_csv(os.path.join(data_folder, csv_file))

    print("Columns:", comments_df.columns)
    logging.info(f"Columns: {comments_df.columns}")

    if "cleaned_comment" in comments_df.columns:
        comments_df["cleaned_comment"] = comments_df["cleaned_comment"].astype(str)
        comments_df = comments_df.dropna(subset=["cleaned_comment"])
        comments = comments_df["cleaned_comment"].tolist()
        print(f"Number of comments: {len(comments)}")
        logging.info(f"Number of comments: {len(comments)}")
    else:
        print(f"Missing 'cleaned_comment' column in {csv_file}. Exiting.")
        logging.error(f"Missing 'cleaned_comment' column in {csv_file}. Exiting.")
        return

    # Create output folder for the artist
    output_folder = f"output_{artist_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Stream results to a file
    stream_file_path = os.path.join(output_folder, f"{artist_name}_analysis_stream.txt")
    stream_file = open(stream_file_path, "w")

    # 2. Feature Extraction
    print("Extracting features using TF-IDF...")
    logging.info("Extracting features using TF-IDF...")
    stream_file.write("Feature Extraction:\n")
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(comments)
    print("Feature extraction completed.")
    logging.info("Feature extraction completed.")
    stream_file.write("Feature extraction completed.\n")

    # Reduce dimensionality using Truncated SVD
    print("Reducing dimensionality using Truncated SVD...")
    logging.info("Reducing dimensionality using Truncated SVD...")
    stream_file.write("Dimensionality Reduction:\n")
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_features = svd.fit_transform(features)
    print("Dimensionality reduction completed.")
    logging.info("Dimensionality reduction completed.")
    stream_file.write("Dimensionality reduction completed.\n")

    # 3. Clustering
    num_clusters = 10
    print(f"Performing clustering with {num_clusters} clusters...")
    logging.info(f"Performing clustering with {num_clusters} clusters...")
    stream_file.write(f"Clustering with {num_clusters} clusters:\n")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(reduced_features)
    clusters = kmeans.labels_
    print("Clustering completed.")
    logging.info("Clustering completed.")
    stream_file.write("Clustering completed.\n")

    # 4. Cluster Analysis
    print("Analyzing clusters...")
    logging.info("Analyzing clusters...")
    stream_file.write("Cluster Analysis:\n")
    cluster_top_words = []
    cluster_comments = []
    for i in range(num_clusters):
        cluster_comments.append(
            [comments[j] for j in range(len(comments)) if clusters[j] == i]
        )
        cluster_vectorizer = TfidfVectorizer(max_features=10)
        cluster_features = cluster_vectorizer.fit_transform(cluster_comments[i])
        top_words = cluster_vectorizer.get_feature_names_out()
        cluster_top_words.append(top_words)
        print(f"Cluster {i+1}: Top Words - {', '.join(top_words)}")
        logging.info(f"Cluster {i+1}: Top Words - {', '.join(top_words)}")
        stream_file.write(f"Cluster {i+1}: Top Words - {', '.join(top_words)}\n")

    # Save cluster analysis to a file
    cluster_analysis_file_path = os.path.join(
        output_folder, f"{artist_name}_cluster_analysis.txt"
    )
    with open(cluster_analysis_file_path, "w") as f:
        for i in range(num_clusters):
            f.write(f"Cluster {i+1}:\n")
            f.write(f"Top Words: {', '.join(cluster_top_words[i])}\n")
            f.write(f"Number of Comments: {len(cluster_comments[i])}\n")
            f.write("Top Comments:\n")
            for comment in cluster_comments[i][:5]:  # Save top 5 comments per cluster
                f.write(f"- {comment}\n")
            f.write("\n")

    print("Cluster analysis saved to file.")
    logging.info("Cluster analysis saved to file.")
    stream_file.write("Cluster analysis saved to file.\n")

    # 5. Graph Analysis
    print("Performing graph analysis...")
    logging.info("Performing graph analysis...")
    stream_file.write("Graph Analysis:\n")
    similarity_threshold = 0.5
    graph = nx.Graph()
    for i in tqdm(range(len(comments)), desc="Building graph"):
        for j in range(i + 1, len(comments)):
            similarity = reduced_features[i].dot(reduced_features[j].T)
            if similarity >= similarity_threshold:
                graph.add_edge(i, j, weight=similarity)

    print("Computing graph metrics...")
    logging.info("Computing graph metrics...")
    stream_file.write("Computing graph metrics...\n")
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph)

    print("Detecting communities...")
    logging.info("Detecting communities...")
    stream_file.write("Detecting communities...\n")
    communities = nx.community.louvain_communities(graph)

    # Save graph analysis to a file
    graph_analysis_file_path = os.path.join(
        output_folder, f"{artist_name}_graph_analysis.txt"
    )
    with open(graph_analysis_file_path, "w") as f:
        f.write("Degree Centrality (Top 10):\n")
        for node, centrality in sorted(
            degree_centrality.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            f.write(f"Comment {node}: {centrality}\n")

        f.write("\nBetweenness Centrality (Top 10):\n")
        for node, centrality in sorted(
            betweenness_centrality.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            f.write(f"Comment {node}: {centrality}\n")

        f.write("\nPageRank (Top 10):\n")
        for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]:
            f.write(f"Comment {node}: {score}\n")

        f.write("\nCommunities:\n")
        for i, community in enumerate(communities, start=1):
            f.write(f"Community {i}:\n")
            f.write(f"Size: {len(community)}\n")
            f.write("Members:\n")
            for member in community:
                f.write(f"- Comment {member}\n")
            f.write("\n")

    print("Graph analysis saved to file.")
    logging.info("Graph analysis saved to file.")
    stream_file.write("Graph analysis saved to file.\n")

    # 6. Sentiment Analysis
    print("Performing sentiment analysis...")
    logging.info("Performing sentiment analysis...")
    stream_file.write("Sentiment Analysis:\n")
    sentiments = []
    for comment in tqdm(comments, desc="Analyzing sentiment"):
        sentiment = TextBlob(comment).sentiment
        sentiments.append(sentiment)

    # Save sentiment analysis to a file
    sentiment_analysis_file_path = os.path.join(
        output_folder, f"{artist_name}_sentiment_analysis.txt"
    )
    with open(sentiment_analysis_file_path, "w") as f:
        f.write("Sentiment Analysis:\n")
        f.write(
            f"Average Polarity: {sum(sentiment.polarity for sentiment in sentiments) / len(sentiments)}\n"
        )
        f.write(
            f"Average Subjectivity: {sum(sentiment.subjectivity for sentiment in sentiments) / len(sentiments)}\n"
        )
        f.write("\nTop Positive Comments:\n")
        for sentiment, comment in sorted(
            zip(sentiments, comments), key=lambda x: x[0].polarity, reverse=True
        )[:5]:
            f.write(f"- {comment}\n")
            f.write(
                f"  Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}\n"
            )
        f.write("\nTop Negative Comments:\n")
        for sentiment, comment in sorted(
            zip(sentiments, comments), key=lambda x: x[0].polarity
        )[:5]:
            f.write(f"- {comment}\n")
            f.write(
                f"  Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}\n"
            )

    print("Sentiment analysis saved to file.")
    logging.info("Sentiment analysis saved to file.")
    stream_file.write("Sentiment analysis saved to file.\n")

    # 7. Keyword Extraction
    print("Extracting keywords...")
    logging.info("Extracting keywords...")
    stream_file.write("Keyword Extraction:\n")
    keyword_vectorizer = TfidfVectorizer(max_features=20)
    keyword_features = keyword_vectorizer.fit_transform(comments)
    keywords = keyword_vectorizer.get_feature_names_out()

    # Save keyword extraction to a file
    keyword_extraction_file_path = os.path.join(
        output_folder, f"{artist_name}_keyword_extraction.txt"
    )
    with open(keyword_extraction_file_path, "w") as f:
        f.write("Top Keywords:\n")
        for keyword in keywords:
            f.write(f"- {keyword}\n")

    print("Keyword extraction saved to file.")
    logging.info("Keyword extraction saved to file.")
    stream_file.write("Keyword extraction saved to file.\n")

    # 8. Topic Modeling
    print("Performing topic modeling...")
    logging.info("Performing topic modeling...")
    stream_file.write("Topic Modeling:\n")
    num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_vectorizer = CountVectorizer(max_features=1000)
    topic_features = topic_vectorizer.fit_transform(comments)
    lda.fit(topic_features)

    # Save topic modeling to a file
    topic_modeling_file_path = os.path.join(
        output_folder, f"{artist_name}_topic_modeling.txt"
    )
    with open(topic_modeling_file_path, "w") as f:
        f.write(f"Number of Topics: {num_topics}\n\n")
        for i, topic in enumerate(lda.components_, start=1):
            top_words = [
                topic_vectorizer.get_feature_names_out()[idx]
                for idx in topic.argsort()[-10:]
            ]
            f.write(f"Topic {i}:\n")
            f.write(f"Top Words: {', '.join(top_words)}\n\n")

    print("Topic modeling saved to file.")
    logging.info("Topic modeling saved to file.")
    stream_file.write("Topic modeling saved to file.\n")

    # 9. Similarity Analysis
    print("Performing similarity analysis...")
    logging.info("Performing similarity analysis...")
    stream_file.write("Similarity Analysis:\n")
    similarity_matrix = cosine_similarity(reduced_features)
    np.fill_diagonal(similarity_matrix, 0)  # Set self-similarity to 0

    # Find the most similar comments
    most_similar_comments = np.unravel_index(
        similarity_matrix.argmax(), similarity_matrix.shape
    )
    comment1, comment2 = most_similar_comments
    similarity_score = similarity_matrix[comment1, comment2]

    # Save similarity analysis to a file
    similarity_analysis_file_path = os.path.join(
        output_folder, f"{artist_name}_similarity_analysis.txt"
    )
    with open(similarity_analysis_file_path, "w") as f:
        f.write("Most Similar Comments:\n")
        f.write(f"Comment {comment1}: {comments[comment1]}\n")
        f.write(f"Comment {comment2}: {comments[comment2]}\n")
        f.write(f"Similarity Score: {similarity_score}\n")

    print("Similarity analysis saved to file.")
    logging.info("Similarity analysis saved to file.")
    stream_file.write("Similarity analysis saved to file.\n")

    # 10. Word Cloud
    print("Generating word cloud...")
    logging.info("Generating word cloud...")
    stream_file.write("Word Cloud Generation:\n")
    text = " ".join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    wordcloud_file_path = os.path.join(output_folder, f"{artist_name}_wordcloud.png")
    plt.savefig(wordcloud_file_path, dpi=300)
    plt.close()
    print("Word cloud saved to file.")
    logging.info("Word cloud saved to file.")
    stream_file.write("Word cloud saved to file.\n")

    # 11. Visualizations
    print("Generating visualizations...")
    logging.info("Generating visualizations...")
    stream_file.write("Visualizations:\n")

    # t-SNE Visualization
    print("Visualizing clusters using t-SNE...")
    logging.info("Visualizing clusters using t-SNE...")
    stream_file.write("t-SNE Visualization:\n")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(reduced_features)
    plt.figure(figsize=(12, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
    for i, color in enumerate(colors):
        plt.scatter(
            tsne_features[clusters == i, 0],
            tsne_features[clusters == i, 1],
            c=[color],
            label=f"Cluster {i+1}",
            alpha=0.8,
        )
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"Cluster Visualization (t-SNE) - {artist_name}")
    plt.colorbar(ticks=range(num_clusters), label="Cluster")
    plt.legend(title="Clusters", loc="upper right")
    plt.tight_layout()
    tsne_file_path = os.path.join(
        output_folder, f"{artist_name}_tsne_visualization.png"
    )
    plt.savefig(tsne_file_path, dpi=300)
    plt.close()
    print("t-SNE visualization saved to file.")
    logging.info("t-SNE visualization saved to file.")
    stream_file.write("t-SNE visualization saved to file.\n")

    # Graph Visualization
    print("Visualizing graph...")
    logging.info("Visualizing graph...")
    stream_file.write("Graph Visualization:\n")
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=10,
        node_color=clusters[: len(graph.nodes())],
        cmap=plt.cm.rainbow,
    )
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.3)
    plt.axis("off")
    plt.title(f"Graph Visualization - {artist_name}")
    plt.tight_layout()
    graph_file_path = os.path.join(
        output_folder, f"{artist_name}_graph_visualization.png"
    )
    plt.savefig(graph_file_path, dpi=300)
    plt.close()
    print("Graph visualization saved to file.")
    logging.info("Graph visualization saved to file.")
    stream_file.write("Graph visualization saved to file.\n")

    # Topic Distribution Visualization
    print("Visualizing topic distribution...")
    logging.info("Visualizing topic distribution...")
    stream_file.write("Topic Distribution Visualization:\n")
    topic_probabilities = lda.transform(topic_features)
    dominant_topics = np.argmax(topic_probabilities, axis=1)
    topic_counts = np.bincount(dominant_topics)
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_topics), topic_counts, align="center", alpha=0.8)
    plt.xticks(range(num_topics), [f"Topic {i+1}" for i in range(num_topics)])
    plt.ylabel("Number of Comments")
    plt.title(f"Topic Distribution - {artist_name}")
    plt.tight_layout()
    topic_distribution_file_path = os.path.join(
        output_folder, f"{artist_name}_topic_distribution.png"
    )
    plt.savefig(topic_distribution_file_path, dpi=300)
    plt.close()
    print("Topic distribution visualization saved to file.")
    logging.info("Topic distribution visualization saved to file.")
    stream_file.write("Topic distribution visualization saved to file.\n")

    # Sentiment Distribution Visualization
    print("Visualizing sentiment distribution...")
    logging.info("Visualizing sentiment distribution...")
    stream_file.write("Sentiment Distribution Visualization:\n")
    sentiment_scores = [sentiment.polarity for sentiment in sentiments]
    plt.figure(figsize=(10, 6))
    plt.hist(sentiment_scores, bins=20, range=(-1, 1), alpha=0.8)
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Number of Comments")
    plt.title(f"Sentiment Distribution - {artist_name}")
    plt.tight_layout()
    sentiment_distribution_file_path = os.path.join(
        output_folder, f"{artist_name}_sentiment_distribution.png"
    )
    plt.savefig(sentiment_distribution_file_path, dpi=300)
    plt.close()
    print("Sentiment distribution visualization saved to file.")
    logging.info("Sentiment distribution visualization saved to file.")
    stream_file.write("Sentiment distribution visualization saved to file.\n")

    print("Analysis completed.")
    logging.info("Analysis completed.")
    stream_file.write("Analysis completed.")

    # Close the stream file
    stream_file.close()


# Main script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the artist name as a command-line argument.")
        print("Usage: python script_name.py 'artist_name'")
        sys.exit(1)

    artist_name = sys.argv[1]
    perform_comment_analysis(artist_name)
