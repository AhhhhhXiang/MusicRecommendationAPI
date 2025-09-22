import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate, Lambda, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from django.db.models import Q
import time
from django.apps import apps

class RecommendationEngine:
    def __init__(self):
        self.base_network = None
        self.recommender = None
        self.song_embeddings = None
        self.track_ids = None
        self.categorical_dims = None
        self.initialized = False
        
    # Build the Best Hyperparameter Network
    def build_advanced_base_network(self, num_numerical_features, categorical_dims):
        BEST_PARAMS = {
            'batch_size': 128,
            'dropout_1': 0.2,
            'dropout_2': 0.3,
            'embedding_dim': 8,
            'hidden_dim_1': 256,
            'hidden_dim_2': 256,
            'l2_reg': 0.01,
            'learning_rate': 0.001
        }
        
        print(f"Building model with optimized hyperparameters: {BEST_PARAMS}")
        print(f"Categorical dimensions: {categorical_dims}")
        
        numerical_input = Input(shape=(num_numerical_features,), name='numerical_input')
        
        categorical_inputs = []
        categorical_embeddings = []
        
        for i, (dim, name) in enumerate(categorical_dims):
            cat_input = Input(shape=(1,), name=f'{name}_input')
            embedding = Embedding(
                input_dim=dim, 
                output_dim=min(BEST_PARAMS['embedding_dim'], dim//2 + 1),
                embeddings_regularizer=l1_l2(l1=0.0, l2=BEST_PARAMS['l2_reg']/10)
            )(cat_input)
            flatten = Flatten()(embedding)
            categorical_inputs.append(cat_input)
            categorical_embeddings.append(flatten)
        
        if categorical_embeddings:
            merged = Concatenate()([numerical_input] + categorical_embeddings)
        else:
            merged = numerical_input
        
        # First hidden layer
        x1 = Dense(BEST_PARAMS['hidden_dim_1'], activation='relu', 
                   kernel_regularizer=l1_l2(l1=0.0, l2=BEST_PARAMS['l2_reg']))(merged)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(BEST_PARAMS['dropout_1'])(x1)
        
        # Second hidden layer (same size for residual connection)
        x2 = Dense(BEST_PARAMS['hidden_dim_2'], activation='relu',
                   kernel_regularizer=l1_l2(l1=0.0, l2=BEST_PARAMS['l2_reg']))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(BEST_PARAMS['dropout_2'])(x2)
        
        # Residual connection
        x2 = Add()([x1, x2])
        
        # Output layer with L2 normalization
        output = Dense(32, activation='linear',
                       kernel_regularizer=l1_l2(l1=0.0, l2=BEST_PARAMS['l2_reg']))(x2)
        output = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(output)
        
        return Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
    
    # Initialize the recommendation engine
    def initialize(self):
        if self.initialized:
            return True
            
        try:
            Track = apps.get_model('nexttrack', 'Track')
            ProcessedTrack = apps.get_model('nexttrack', 'ProcessedTrack')
            TrackEmbedding = apps.get_model('nexttrack', 'TrackEmbedding')
            
            # Check if embeddings exist
            if TrackEmbedding.objects.count() == 0:
                print("Generating embeddings with optimized hyperparameters...")
                self._generate_and_store_embeddings(Track, ProcessedTrack, TrackEmbedding)
            else:
                print("Loading embeddings from database...")
                self._load_embeddings_from_database(TrackEmbedding)
            
            self._build_recommender()
            self.initialized = True
            print(f"✅ Recommendation engine initialized with {len(self.track_ids)} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing recommendation engine: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Helper Function - Generate and store embeddings
    def _generate_and_store_embeddings(self, Track, ProcessedTrack, TrackEmbedding):
        processed_tracks = ProcessedTrack.objects.select_related('track').all()
        
        numerical_features = []
        categorical_features = []
        track_ids = []
        
        # First, find the maximum values for each categorical feature to determine dimensions
        max_key = 0
        max_mode = 0
        max_ts = 0
        
        for pt in processed_tracks:
            # Collect features
            numerical_features.append([
                pt.popularity_scaled or 0,
                pt.duration_ms_scaled or 0,
                pt.danceability_scaled or 0,
                pt.energy_scaled or 0,
                pt.loudness_scaled or 0,
                pt.speechiness_scaled or 0,
                pt.acousticness_scaled or 0,
                pt.instrumentalness_scaled or 0,
                pt.liveness_scaled or 0,
                pt.valence_scaled or 0,
                pt.tempo_scaled or 0
            ])
            categorical_features.append([
                pt.key_encoded or 0,
                pt.mode_encoded or 0,
                pt.time_signature_encoded or 0
            ])
            track_ids.append(pt.track_id)
            
            # Update maximum values
            max_key = max(max_key, pt.key_encoded or 0)
            max_mode = max(max_mode, pt.mode_encoded or 0)
            max_ts = max(max_ts, pt.time_signature_encoded or 0)
        
        X_numerical = np.array(numerical_features, dtype=np.float32)
        X_categorical = np.array(categorical_features, dtype=np.int32)
        
        # Use dynamic dimensions based on actual data (add 1 because encoding starts at 0)
        key_dim = max_key + 1
        mode_dim = max_mode + 1
        ts_dim = max_ts + 1
        
        print(f"Dynamic categorical dimensions based on data:")
        print(f"  Key: {key_dim} (max value: {max_key})")
        print(f"  Mode: {mode_dim} (max value: {max_mode})")
        print(f"  Time Signature: {ts_dim} (max value: {max_ts})")
        
        self.categorical_dims = [
            (max(key_dim, 1), 'key'),      # Ensure at least dimension 1
            (max(mode_dim, 1), 'mode'),
            (max(ts_dim, 1), 'time_signature')
        ]
        
        self.base_network = self.build_advanced_base_network(
            X_numerical.shape[1], self.categorical_dims
        )
        
        optimizer = Adam(learning_rate=0.001)
        self.base_network.compile(optimizer=optimizer, loss='mse')
        
        embeddings = self._generate_embeddings(X_numerical, X_categorical)
        self._store_embeddings(track_ids, embeddings, TrackEmbedding)
        
        self.track_ids = track_ids
        self.song_embeddings = embeddings
    
    # Generate embeddings
    def _generate_embeddings(self, X_numerical, X_categorical):
        numerical_inputs = X_numerical
        categorical_inputs = [X_categorical[:, i] for i in range(X_categorical.shape[1])]
        
        inputs = [numerical_inputs] + categorical_inputs
        
        print("Generating embeddings...")
        print(f"Numerical shape: {X_numerical.shape}")
        print(f"Categorical shapes: {[arr.shape for arr in categorical_inputs]}")
        
        # Verify categorical values are within expected ranges
        for i, (dim, name) in enumerate(self.categorical_dims):
            cat_values = categorical_inputs[i]
            min_val = cat_values.min()
            max_val = cat_values.max()
            print(f"{name} values range: {min_val} to {max_val} (expected: 0 to {dim-1})")
            
            if max_val >= dim:
                print(f"⚠️  WARNING: {name} value {max_val} exceeds expected maximum {dim-1}")
                # Clip values to valid range
                categorical_inputs[i] = np.clip(cat_values, 0, dim - 1)
                print(f"Clipped {name} values to valid range")
        
        return self.base_network.predict(inputs, batch_size=128, verbose=1)
    
    # Helper Function - Store embeddings in database
    def _store_embeddings(self, track_ids, embeddings, TrackEmbedding):
        Track = apps.get_model('nexttrack', 'Track')
        embeddings_to_create = []
        
        for track_id, embedding in zip(track_ids, embeddings):
            try:
                track = Track.objects.get(track_id=track_id)
                embedding_data = {
                    'track': track,
                    **{f'embedding_{i}': float(embedding[i]) for i in range(32)}
                }
                embeddings_to_create.append(TrackEmbedding(**embedding_data))
            except Track.DoesNotExist:
                continue
        
        TrackEmbedding.objects.bulk_create(embeddings_to_create, batch_size=1000)
        print(f"Stored {len(embeddings_to_create)} embeddings")
    
    # Load embeddings from database
    def _load_embeddings_from_database(self, TrackEmbedding):
        embeddings_qs = TrackEmbedding.objects.select_related('track').all()
        
        self.track_ids = []
        self.song_embeddings = []
        
        for embedding_obj in embeddings_qs:
            self.track_ids.append(embedding_obj.track_id)
            embedding_vector = np.array([
                getattr(embedding_obj, f'embedding_{i}') for i in range(32)
            ], dtype=np.float32)
            self.song_embeddings.append(embedding_vector)
        
        self.song_embeddings = np.array(self.song_embeddings)
        self.track_ids = np.array(self.track_ids)
    
    # Build nearest neighbors recommender
    def _build_recommender(self):
        self.recommender = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute')
        self.recommender.fit(self.song_embeddings)
        print("Recommender model built successfully!")
    
    # Get recommendations for one or more tracks with optional preference filtering
    def get_recommendations(self, track_id, n_recommendations=5, diversity_factor=0.1, preferences=None):
        """
        Args:
            track_id: Single track_id (str) or list of track_ids (list)
            n_recommendations: Number of recommendations to return
            diversity_factor: Factor to promote diversity (0.0-1.0)
            preferences: Optional preference filters
        
        Returns:
            List of recommendations: [{'track_id': str, 'similarity': float}]
        """
        if not self.initialized:
            if not self.initialize():
                return None
        
        # Handle both single track_id and list of track_ids
        if isinstance(track_id, str):
            track_ids = [track_id]
        elif isinstance(track_id, list):
            track_ids = track_id
        else:
            raise ValueError("track_id must be a string or list of strings")
        
        try:
            # Find indices for all input tracks
            track_indices = []
            for t_id in track_ids:
                idx = np.where(self.track_ids == t_id)[0]
                if len(idx) > 0:
                    track_indices.append(idx[0])
            
            if not track_indices:
                return None
            
            # Calculate average embedding for multiple tracks
            if len(track_indices) == 1:
                # Single track - use its embedding directly
                query_embedding = self.song_embeddings[track_indices[0]].reshape(1, -1)
            else:
                # Multiple tracks - average their embeddings
                track_embeddings = self.song_embeddings[track_indices]
                query_embedding = np.mean(track_embeddings, axis=0).reshape(1, -1)
                # Normalize the average embedding
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Get more candidates to allow for preference filtering
            n_candidates = max(n_recommendations * 5, 50) if preferences else n_recommendations + 20
            distances, indices = self.recommender.kneighbors(
                query_embedding, n_neighbors=min(n_candidates, len(self.track_ids))
            )
            
            # Get all candidate tracks (excluding input tracks)
            candidate_recommendations = []
            for i, idx in enumerate(indices[0]):
                candidate_track_id = self.track_ids[idx]
                
                # Skip if candidate is one of the input tracks
                if candidate_track_id in track_ids:
                    continue
                    
                similarity = 1 - distances[0][i]
                candidate_recommendations.append({
                    'track_id': candidate_track_id,
                    'similarity': float(similarity),
                    'index': idx
                })
            
            # Apply diversity selection
            selected_indices = self._apply_diversity_selection(
                [cand['index'] for cand in candidate_recommendations],
                distances[0],
                n_candidates,
                diversity_factor
            )
            
            # Get track objects for preference filtering if needed
            recommendations = []
            if preferences:
                # Get track objects for selected candidates
                selected_track_ids = [self.track_ids[idx] for idx in selected_indices]
                Track = apps.get_model('nexttrack', 'Track')
                selected_tracks = {t.track_id: t for t in Track.objects.filter(track_id__in=selected_track_ids)}
                
                # Prepare recommendations with track objects for filtering
                for cand in candidate_recommendations:
                    if cand['index'] in selected_indices:
                        track = selected_tracks.get(cand['track_id'])
                        if track:
                            recommendations.append({
                                'track': track,
                                'similarity': cand['similarity']
                            })
                
                # Apply preference filtering
                from .preferences import apply_preferences_to_recommendations
                recommendations = apply_preferences_to_recommendations(recommendations, preferences)
                
                # Convert back to track_id format for consistency
                filtered_recommendations = []
                for rec in recommendations[:n_recommendations]:
                    filtered_recommendations.append({
                        'track_id': rec['track'].track_id,
                        'similarity': rec['similarity']
                    })
                
                return filtered_recommendations
            
            else:
                # Original behavior without preferences
                for cand in candidate_recommendations:
                    if cand['index'] in selected_indices:
                        recommendations.append({
                            'track_id': cand['track_id'],
                            'similarity': cand['similarity']
                        })
                        
                        if len(recommendations) >= n_recommendations:
                            break
                
                return recommendations
                
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return None
    
    # Apply diversity-aware selection
    def _apply_diversity_selection(self, indices, distances, n_recommendations, diversity_factor):
        if diversity_factor == 0 or len(indices) <= n_recommendations:
            return indices[1:n_recommendations + 1]
        
        selected = [indices[1]]
        remaining = list(indices[2:])
        
        while len(selected) < n_recommendations and remaining:
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in remaining:
                candidate_idx = np.where(indices == candidate)[0][0]
                similarity_score = 1 - distances[candidate_idx]
                
                if len(selected) > 0:
                    candidate_embedding = self.song_embeddings[candidate]
                    selected_embeddings = self.song_embeddings[selected]
                    similarities = [
                        np.dot(candidate_embedding, sel_emb) 
                        for sel_emb in selected_embeddings
                    ]
                    diversity_score = 1 - np.mean(similarities)
                else:
                    diversity_score = 0
                
                combined_score = (1 - diversity_factor) * similarity_score + diversity_factor * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected

recommendation_engine = RecommendationEngine()