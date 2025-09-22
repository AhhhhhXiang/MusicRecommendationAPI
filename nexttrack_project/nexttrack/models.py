# nexttrack/models.py
from django.db import models

class Track(models.Model):
    track_id = models.CharField(max_length=255, primary_key=True)
    artists = models.CharField(max_length=1000, null=True, blank=True)
    album_name = models.CharField(max_length=1000, null=True, blank=True)
    track_name = models.CharField(max_length=1000, null=True, blank=True)
    popularity = models.IntegerField(null=True, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)
    explicit = models.BooleanField(null=True, blank=True)
    danceability = models.FloatField(null=True, blank=True)
    energy = models.FloatField(null=True, blank=True)
    key = models.CharField(max_length=10, null=True, blank=True)
    loudness = models.FloatField(null=True, blank=True)
    mode = models.IntegerField(null=True, blank=True)
    speechiness = models.FloatField(null=True, blank=True)
    acousticness = models.FloatField(null=True, blank=True)
    instrumentalness = models.FloatField(null=True, blank=True)
    liveness = models.FloatField(null=True, blank=True)
    valence = models.FloatField(null=True, blank=True)
    tempo = models.FloatField(null=True, blank=True)
    time_signature = models.IntegerField(null=True, blank=True)
    track_genre = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'track'

    def __str__(self):
        return f"{self.track_name} by {self.artists}"


class ProcessedTrack(models.Model):
    track = models.OneToOneField(Track, on_delete=models.CASCADE, primary_key=True)
    popularity_scaled = models.FloatField(null=True, blank=True)
    duration_ms_scaled = models.FloatField(null=True, blank=True)
    danceability_scaled = models.FloatField(null=True, blank=True)
    energy_scaled = models.FloatField(null=True, blank=True)
    loudness_scaled = models.FloatField(null=True, blank=True)
    speechiness_scaled = models.FloatField(null=True, blank=True)
    acousticness_scaled = models.FloatField(null=True, blank=True)
    instrumentalness_scaled = models.FloatField(null=True, blank=True)
    liveness_scaled = models.FloatField(null=True, blank=True)
    valence_scaled = models.FloatField(null=True, blank=True)
    tempo_scaled = models.FloatField(null=True, blank=True)
    key_encoded = models.IntegerField(null=True, blank=True)
    mode_encoded = models.IntegerField(null=True, blank=True)
    time_signature_encoded = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'processed_track'


class TrackEmbedding(models.Model):
    track = models.OneToOneField(Track, on_delete=models.CASCADE, primary_key=True)
    # Create all 32 embedding fields
    embedding_0 = models.FloatField(null=True, blank=True)
    embedding_1 = models.FloatField(null=True, blank=True)
    embedding_2 = models.FloatField(null=True, blank=True)
    embedding_3 = models.FloatField(null=True, blank=True)
    embedding_4 = models.FloatField(null=True, blank=True)
    embedding_5 = models.FloatField(null=True, blank=True)
    embedding_6 = models.FloatField(null=True, blank=True)
    embedding_7 = models.FloatField(null=True, blank=True)
    embedding_8 = models.FloatField(null=True, blank=True)
    embedding_9 = models.FloatField(null=True, blank=True)
    embedding_10 = models.FloatField(null=True, blank=True)
    embedding_11 = models.FloatField(null=True, blank=True)
    embedding_12 = models.FloatField(null=True, blank=True)
    embedding_13 = models.FloatField(null=True, blank=True)
    embedding_14 = models.FloatField(null=True, blank=True)
    embedding_15 = models.FloatField(null=True, blank=True)
    embedding_16 = models.FloatField(null=True, blank=True)
    embedding_17 = models.FloatField(null=True, blank=True)
    embedding_18 = models.FloatField(null=True, blank=True)
    embedding_19 = models.FloatField(null=True, blank=True)
    embedding_20 = models.FloatField(null=True, blank=True)
    embedding_21 = models.FloatField(null=True, blank=True)
    embedding_22 = models.FloatField(null=True, blank=True)
    embedding_23 = models.FloatField(null=True, blank=True)
    embedding_24 = models.FloatField(null=True, blank=True)
    embedding_25 = models.FloatField(null=True, blank=True)
    embedding_26 = models.FloatField(null=True, blank=True)
    embedding_27 = models.FloatField(null=True, blank=True)
    embedding_28 = models.FloatField(null=True, blank=True)
    embedding_29 = models.FloatField(null=True, blank=True)
    embedding_30 = models.FloatField(null=True, blank=True)
    embedding_31 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'track_embedding'