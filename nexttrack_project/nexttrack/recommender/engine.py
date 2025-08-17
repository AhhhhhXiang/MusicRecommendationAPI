import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class TrackRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(inplace=True)
        self.features = self.df[['danceability', 'energy', 'key', 'loudness',
                                 'speechiness', 'acousticness', 'instrumentalness', 'tempo']]
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.df['cluster'] = self.kmeans.fit_predict(self.features)

    def recommend(self, track_ids, top_n=5):
        if isinstance(track_ids, str):
            track_ids = [track_ids]

        # Filter valid track IDs only
        input_tracks = self.df[self.df['track_id'].isin(track_ids)]
        if input_tracks.empty:
            return []

        # Determine cluster mode (majority cluster)
        cluster_counts = input_tracks['cluster'].value_counts()
        primary_cluster = cluster_counts.idxmax()

        # Filter tracks in the same cluster
        similar_tracks = self.df[self.df['cluster'] == primary_cluster].copy()

        # Average the feature vectors of the input tracks
        avg_vector = input_tracks[self.features.columns].mean().values.reshape(1, -1)

        # Compute cosine similarity
        similarities = cosine_similarity(avg_vector, similar_tracks[self.features.columns])
        similar_tracks['similarity'] = similarities[0]

        # Exclude input track IDs from the results
        results = similar_tracks[~similar_tracks['track_id'].isin(track_ids)]
        results = results.sort_values(by='similarity', ascending=False).head(top_n)

        return results.to_dict(orient='records')