import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain


def analyze_graph(artist_graph):
    # Community Detection (Louvain)
    partition = community_louvain.best_partition(artist_graph)
    print("\nCommunities:")
    for artist, community_id in partition.items():
        print(f"{artist}: Community {community_id}")

    # Centrality Measures
    degree_centrality = nx.degree_centrality(artist_graph)
    betweenness_centrality = nx.betweenness_centrality(artist_graph)
    eigenvector_centrality = nx.eigenvector_centrality(artist_graph)

    print("\nTop 5 Artists by Degree Centrality:")
    for artist, centrality in sorted(
        degree_centrality.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{artist}: {centrality:.3f}")

    print("\nTop 5 Artists by Betweenness Centrality:")
    for artist, centrality in sorted(
        betweenness_centrality.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{artist}: {centrality:.3f}")

    print("\nTop 5 Artists by Eigenvector Centrality:")
    for artist, centrality in sorted(
        eigenvector_centrality.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{artist}: {centrality:.3f}")

    # PageRank
    pagerank = nx.pagerank(artist_graph)
    print("\nTop 5 Artists by PageRank:")
    for artist, rank in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{artist}: {rank:.3f}")


def load_and_compare_all_artists():
    # Assuming the current directory has a 'data' folder with comment files
    data_folder = os.path.join(os.getcwd(), "data")
    comment_files = [
        f
        for f in os.listdir(data_folder)
        if f.startswith("cleaned_") and f.endswith("_comments.csv")
    ]

    all_comments = pd.DataFrame()
    artist_names = []

    # Load comments for each artist and combine them into a single DataFrame
    for file in comment_files:
        artist_name = file.split("_")[1]  # Extract the artist's name from the file name
        artist_comments = pd.read_csv(
            os.path.join(data_folder, file), header=None, names=["comment"]
        )
        artist_comments["artist"] = artist_name
        all_comments = pd.concat([all_comments, artist_comments], ignore_index=True)
        artist_names.append(artist_name)

    # Prepare the data
    all_comments["comment"] = all_comments["comment"].fillna("")
    all_comments = all_comments[
        all_comments["comment"].apply(lambda x: isinstance(x, str))
    ]

    # Create a TF-IDF vectorizer and transform the comments
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_comments["comment"])

    # Perform clustering on the comments
    kmeans = KMeans(n_clusters=10)  # Adjust the number of clusters as needed
    labels = kmeans.fit_predict(tfidf_matrix)
    all_comments["cluster"] = labels

    # Create the similarity graph
    G = nx.Graph()
    for artist in artist_names:
        G.add_node(artist)

    # Calculate the cosine similarity and update the graph
    for i in range(len(artist_names)):
        for j in range(i + 1, len(artist_names)):
            artist1 = artist_names[i]
            artist2 = artist_names[j]
            artist1_comments = tfidf_matrix[all_comments["artist"] == artist1]
            artist2_comments = tfidf_matrix[all_comments["artist"] == artist2]
            similarity_score = cosine_similarity(
                artist1_comments, artist2_comments
            ).mean()
            G.add_edge(artist1, artist2, weight=similarity_score)

    return G, all_comments


def main():
    # Call the function to load all comments and create the similarity graph
    artist_graph, comments_df = load_and_compare_all_artists()

    # Print out the edges of the graph with the similarity weights
    for edge in artist_graph.edges(data=True):
        print(f"{edge[0]} <--> {edge[1]}, similarity: {edge[2]['weight']}")

    # Draw the graph using NetworkX's drawing tools
    pos = nx.spring_layout(artist_graph)  # Position the nodes using the spring layout
    nx.draw(artist_graph, pos, with_labels=True, font_weight="bold")
    analyze_graph(artist_graph)

    # Save the graph to a PNG file
    plt.savefig("fanbase_similarity_graph.png")
    print("The graph has been saved as 'fanbase_similarity_graph.png'.")

    # Print a sample of the DataFrame to check the comments and clusters
    print("Sample of the comments DataFrame with clusters:")
    print(comments_df.sample(5))


if __name__ == "__main__":
    main()
