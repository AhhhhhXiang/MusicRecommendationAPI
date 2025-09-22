from django.apps import apps

# Get all genres from the database
def get_available_genres():
    try:
        Track = apps.get_model('nexttrack', 'Track')
        # Get distinct genres from the database
        genres = Track.objects.exclude(track_genre__isnull=True)\
                             .exclude(track_genre='')\
                             .values_list('track_genre', flat=True)\
                             .distinct()\
                             .order_by('track_genre')
        return sorted(list(set(genre.lower() for genre in genres if genre)))
    except:
        return []

# Check if a track matches user preferences
def check_track_preferences(track, preferences):
    if not preferences:
        return True
    
    track_genre = (track.track_genre or '').lower().strip()
    
    # Genre preference - exact match
    if 'genres' in preferences and preferences['genres']:
        preferred_genres = [g.lower().strip() for g in preferences['genres']]
        
        # Check if track genre matches any preferred genre
        if not any(pref_genre == track_genre for pref_genre in preferred_genres):
            return False
    
    # Energy level preference
    if 'energy_level' in preferences:
        energy = track.energy or 0
        energy_level = preferences['energy_level'].lower()
        
        if energy_level == 'low' and energy > 0.4:
            return False
        elif energy_level == 'high' and energy < 0.6:
            return False
        elif energy_level == 'medium' and not (0.4 <= energy <= 0.7):
            return False
    
    # Mood preference (based on valence)
    if 'mood' in preferences:
        valence = track.valence or 0
        mood = preferences['mood'].lower()
        
        if mood == 'happy' and valence < 0.6:
            return False
        elif mood == 'sad' and valence > 0.4:
            return False
        elif mood == 'neutral' and not (0.4 <= valence <= 0.6):
            return False
    
    # Tempo preference (BPM)
    if 'tempo_range' in preferences:
        tempo = track.tempo or 0
        min_tempo, max_tempo = preferences['tempo_range']
        
        if tempo < min_tempo or tempo > max_tempo:
            return False
    
    # Duration preference (milliseconds)
    if 'max_duration_ms' in preferences:
        duration_ms = track.duration_ms or 0
        max_duration = preferences['max_duration_ms']
        
        if duration_ms > max_duration:
            return False
    
    # Explicit content preference
    if 'allow_explicit' in preferences and not preferences['allow_explicit']:
        if track.explicit:
            return False
    
    # Popularity preference
    if 'min_popularity' in preferences:
        popularity = track.popularity or 0
        min_popularity = preferences['min_popularity']
        
        if popularity < min_popularity:
            return False
    
    return True

# Filter recommendations based on user preferences
def apply_preferences_to_recommendations(recommendations, preferences):
    if not preferences:
        return recommendations
    
    filtered_recommendations = []
    
    for rec in recommendations:
        if check_track_preferences(rec['track'], preferences):
            filtered_recommendations.append(rec)
    
    return filtered_recommendations


# Preference presets (genres will be validated against database)
PREFERENCE_PRESETS = {
    'energetic_dance': {
        'energy_level': 'high',
        'genres': ['dance', 'edm', 'electronic', 'house', 'techno', 'trance', 'disco'],
        'mood': 'happy',
        'tempo_range': [110, 180]
    },
    'chill_relax': {
        'energy_level': 'low',
        'genres': ['acoustic', 'ambient', 'chill', 'jazz', 'classical', 'soul'],
        'mood': 'neutral',
        'tempo_range': [60, 100]
    },
    'workout': {
        'energy_level': 'high',
        'genres': ['hip-hop', 'rock', 'electronic', 'pop', 'metal'],
        'tempo_range': [120, 200],
        'mood': 'happy'
    },
    'focus_study': {
        'energy_level': 'low',
        'genres': ['classical', 'ambient', 'jazz', 'study'],
        'mood': 'neutral',
        'allow_explicit': False
    },
    'rock_alternative': {
        'genres': ['rock', 'alternative', 'indie', 'punk', 'hard-rock', 'grunge'],
        'energy_level': 'medium',
        'mood': 'neutral'
    },
    'hiphop_rnb': {
        'genres': ['hip-hop', 'r-n-b', 'reggaeton'],
        'energy_level': 'medium',
        'tempo_range': [70, 120]
    },
    'party': {
        'energy_level': 'high',
        'genres': ['pop', 'dance', 'electronic', 'hip-hop', 'reggaeton', 'latin'],
        'mood': 'happy',
        'tempo_range': [100, 160]
    }
}


def validate_genres(genres):
    """Validate if provided genres exist in the database"""
    available_genres = get_available_genres()
    invalid_genres = []
    valid_genres = []
    
    for genre in genres:
        if genre.lower() in available_genres:
            valid_genres.append(genre.lower())
        else:
            invalid_genres.append(genre)
    
    return valid_genres, invalid_genres


def filter_valid_preset_genres(preset_name):
    """Filter preset genres to only include those that exist in database"""
    available_genres = get_available_genres()
    preset = PREFERENCE_PRESETS.get(preset_name, {})
    
    if 'genres' in preset:
        valid_genres = [g for g in preset['genres'] if g in available_genres]
        return {**preset, 'genres': valid_genres}
    
    return preset