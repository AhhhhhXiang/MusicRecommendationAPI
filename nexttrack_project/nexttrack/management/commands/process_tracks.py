from django.core.management.base import BaseCommand
from nexttrack.models import Track, ProcessedTrack
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class Command(BaseCommand):
    help = 'Process raw tracks and create ProcessedTrack records'
    
    def handle(self, *args, **options):
        self.stdout.write('Processing tracks...')
        
        # Check if we have tracks
        track_count = Track.objects.count()
        if track_count == 0:
            self.stdout.write(
                self.style.ERROR('No Track records found. Please import tracks first.')
            )
            return
        
        # Delete existing processed tracks
        ProcessedTrack.objects.all().delete()
        
        # Get all tracks
        tracks = Track.objects.all()
        
        # Prepare data for scaling
        numerical_features = []
        categorical_features = []
        track_ids = []
        
        for track in tracks:
            numerical_features.append([
                track.popularity or 0,
                track.duration_ms or 0,
                track.danceability or 0,
                track.energy or 0,
                track.loudness or 0,
                track.speechiness or 0,
                track.acousticness or 0,
                track.instrumentalness or 0,
                track.liveness or 0,
                track.valence or 0,
                track.tempo or 0
            ])
            
            # Handle categorical data with defaults
            key_val = str(track.key) if track.key is not None else '0'
            mode_val = str(track.mode) if track.mode is not None else '0'
            ts_val = str(track.time_signature) if track.time_signature is not None else '4'
            
            categorical_features.append([key_val, mode_val, ts_val])
            track_ids.append(track.track_id)
        
        # Scale numerical features
        numerical_features = np.array(numerical_features, dtype=np.float32)
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(numerical_features)
        
        # Encode categorical features with LabelEncoder but ensure proper ranges
        key_encoder = LabelEncoder()
        mode_encoder = LabelEncoder()
        ts_encoder = LabelEncoder()
        
        keys = [cat[0] for cat in categorical_features]
        modes = [cat[1] for cat in categorical_features]
        time_signatures = [cat[2] for cat in categorical_features]
        
        # Fit the encoders
        keys_encoded = key_encoder.fit_transform(keys)
        modes_encoded = mode_encoder.fit_transform(modes)
        ts_encoded = ts_encoder.fit_transform(time_signatures)
        
        # Log the ranges to debug
        self.stdout.write(f"Key classes: {key_encoder.classes_}")
        self.stdout.write(f"Mode classes: {mode_encoder.classes_}")
        self.stdout.write(f"Time signature classes: {ts_encoder.classes_}")
        
        self.stdout.write(f"Key encoded range: {keys_encoded.min()} to {keys_encoded.max()}")
        self.stdout.write(f"Mode encoded range: {modes_encoded.min()} to {modes_encoded.max()}")
        self.stdout.write(f"Time signature encoded range: {ts_encoded.min()} to {ts_encoded.max()}")
        
        # Create ProcessedTrack records
        processed_tracks = []
        for i, track_id in enumerate(track_ids):
            try:
                track = Track.objects.get(track_id=track_id)
                processed_track = ProcessedTrack(
                    track=track,
                    popularity_scaled=float(scaled_numerical[i][0]),
                    duration_ms_scaled=float(scaled_numerical[i][1]),
                    danceability_scaled=float(scaled_numerical[i][2]),
                    energy_scaled=float(scaled_numerical[i][3]),
                    loudness_scaled=float(scaled_numerical[i][4]),
                    speechiness_scaled=float(scaled_numerical[i][5]),
                    acousticness_scaled=float(scaled_numerical[i][6]),
                    instrumentalness_scaled=float(scaled_numerical[i][7]),
                    liveness_scaled=float(scaled_numerical[i][8]),
                    valence_scaled=float(scaled_numerical[i][9]),
                    tempo_scaled=float(scaled_numerical[i][10]),
                    key_encoded=int(keys_encoded[i]),
                    mode_encoded=int(modes_encoded[i]),
                    time_signature_encoded=int(ts_encoded[i])
                )
                processed_tracks.append(processed_track)
                
                if len(processed_tracks) % 1000 == 0:
                    self.stdout.write(f'Processed {len(processed_tracks)} tracks...')
                    
            except Track.DoesNotExist:
                continue
            except Exception as e:
                self.stdout.write(f'Error processing track {track_id}: {str(e)}')
                continue
        
        # Bulk create
        if processed_tracks:
            ProcessedTrack.objects.bulk_create(processed_tracks, batch_size=1000)
            self.stdout.write(
                self.style.SUCCESS(f'Processed {len(processed_tracks)} tracks!')
            )
        else:
            self.stdout.write(
                self.style.ERROR('No tracks were processed successfully.')
            )