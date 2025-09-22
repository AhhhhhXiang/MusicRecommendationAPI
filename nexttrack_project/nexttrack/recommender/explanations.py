import numpy as np

# Generate explanation for why a song was recommended
def generate_recommendation_explanation(input_track, recommended_track, similarity, include_explanation=True):
    """
    Args:
        input_track: The original track object
        recommended_track: The recommended track object  
        similarity: Similarity score (0-1)
        include_explanation: Whether to include detailed explanation
    """
    if not include_explanation:
        return None
    
    explanations = []
    
    # Similarity-based explanation
    if similarity > 0.8:
        explanations.append(f"Very high similarity ({similarity:.3f})")
    elif similarity > 0.6:
        explanations.append(f"Similar musical patterns ({similarity:.3f})")
    elif similarity > 0.4:
        explanations.append(f"Moderate similarity ({similarity:.3f})")
    else:
        explanations.append(f"Complementary features ({similarity:.3f})")
    
    # Genre-based explanation
    if input_track.track_genre and recommended_track.track_genre:
        if input_track.track_genre == recommended_track.track_genre:
            explanations.append(f"same genre ({input_track.track_genre})")
    
    # Key audio feature explanations
    feature_comparisons = [
        ('energy', 'similar energy', 0.2),
        ('valence', 'matching mood', 0.2),
        ('danceability', 'comparable danceability', 0.2),
        ('tempo', 'similar tempo', 20),
    ]
    
    for feature, message, threshold in feature_comparisons:
        input_val = getattr(input_track, feature, 0) or 0
        rec_val = getattr(recommended_track, feature, 0) or 0
        
        if feature == 'tempo':
            if abs(input_val - rec_val) < threshold:
                explanations.append(message)
        else:
            if abs(input_val - rec_val) < threshold:
                explanations.append(message)
    
    return ", ".join(explanations[:3])  # Limit to top 3 explanations

# Compare audio features between two tracks
def compare_audio_features(input_track, recommended_track, include_comparison=True):
    if not include_comparison:
        return None
    
    features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness']
    
    comparison = {}
    for feature in features:
        input_val = getattr(input_track, feature, 0) or 0
        rec_val = getattr(recommended_track, feature, 0) or 0
        diff = abs(input_val - rec_val)
        
        comparison[feature] = {
            'input': round(float(input_val), 3),
            'recommended': round(float(rec_val), 3),
            'difference': round(float(diff), 3),
            'similar': diff < 0.2 if feature != 'tempo' else diff < 20
        }
    
    return comparison