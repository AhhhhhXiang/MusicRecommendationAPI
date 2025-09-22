from rest_framework import serializers
from .models import Track
from .recommender.preferences import PREFERENCE_PRESETS, get_available_genres, validate_genres

class TrackSerializer(serializers.ModelSerializer):
    artist = serializers.CharField(source='artists')  # Map 'artists' to 'artist'
    genre = serializers.CharField(source='track_genre')  # Map 'track_genre' to 'genre'
    
    class Meta:
        model = Track
        fields = [
            'track_id', 'track_name', 'artist', 'genre',
            'danceability', 'energy', 'key', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'tempo'
        ]

class RecommendedTrackResponseSerializer(serializers.Serializer):
    recommended_track = TrackSerializer()
    explanation = serializers.CharField()

class RecommendationResponseSerializer(serializers.Serializer):
    input_track = TrackSerializer()
    recommendations = RecommendedTrackResponseSerializer(many=True)

# Track Details
class TrackDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Track
        fields = [
            'track_id', 'artists', 'album_name', 'track_name', 'popularity',
            'duration_ms', 'explicit', 'danceability', 'energy', 'key',
            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'
        ]

class PreferenceSerializer(serializers.Serializer):
    genres = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )
    energy_level = serializers.ChoiceField(
        choices=['low', 'medium', 'high'],
        required=False
    )
    mood = serializers.ChoiceField(
        choices=['sad', 'neutral', 'happy'],
        required=False
    )
    tempo_range = serializers.ListField(
        child=serializers.IntegerField(min_value=0, max_value=300),
        required=False,
        min_length=2,
        max_length=2
    )
    max_duration_ms = serializers.IntegerField(
        min_value=0,
        max_value=3600000,
        required=False
    )
    min_popularity = serializers.IntegerField(
        min_value=0,
        max_value=100,
        required=False
    )
    allow_explicit = serializers.BooleanField(
        default=True,
        required=False
    )
    
    # Validate that provided genres exist in the database
    def validate_genres(self, value):
        if value:
            valid_genres, invalid_genres = validate_genres(value)
            if invalid_genres:
                available_genres = get_available_genres()
                raise serializers.ValidationError(
                    f"Invalid genres: {invalid_genres}. Available genres: {available_genres}"
                )
            return valid_genres
        return value
    
    # Validate tempo range format
    def validate_tempo_range(self, value):
        if value and len(value) == 2:
            if value[0] >= value[1]:
                raise serializers.ValidationError("Tempo range must be [min, max] with min < max")
            if value[0] < 0 or value[1] > 300:
                raise serializers.ValidationError("Tempo range must be between 0 and 300 BPM")
        return value

class RecommendationRequestSerializer(serializers.Serializer):
    track_id = serializers.CharField(required=False)
    track_ids = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )
    n_recommendations = serializers.IntegerField(default=5, min_value=1, max_value=20)
    diversity_factor = serializers.FloatField(default=0.1, min_value=0.0, max_value=1.0)
    explanation_detail = serializers.BooleanField(default=False, required=False)
    preferences = PreferenceSerializer(required=False)
    preference_preset = serializers.ChoiceField(
        choices=list(PREFERENCE_PRESETS.keys()),
        required=False
    )
    
    # Validate that either track_id or track_ids is provided
    def validate(self, data):
        if not data.get('track_id') and not data.get('track_ids'):
            raise serializers.ValidationError("Either 'track_id' or 'track_ids' must be provided")
        
        if data.get('track_id') and data.get('track_ids'):
            raise serializers.ValidationError("Provide either 'track_id' or 'track_ids', not both")
        
        return data