from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Q
from django.http import Http404
from django.shortcuts import get_object_or_404

from .models import Track
from .recommender.engine import recommendation_engine
from .recommender.explanations import generate_recommendation_explanation
from .serializers import TrackSerializer, RecommendationRequestSerializer, TrackDetailSerializer
from .recommender.preferences import PREFERENCE_PRESETS

# View for getting next track recommendations"
class NextTrackView(APIView):
    
    def post(self, request):
        serializer = RecommendationRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Get either single track_id or list of track_ids
        track_id = serializer.validated_data.get('track_id')
        track_ids = serializer.validated_data.get('track_ids')
        
        # Use either single track_id or list of track_ids
        input_tracks = track_ids if track_ids else [track_id]
        
        n_recommendations = serializer.validated_data.get('n_recommendations', 5)
        diversity_factor = serializer.validated_data.get('diversity_factor', 0.1)
        explanation_detail = serializer.validated_data.get('explanation_detail', False)
        preferences = serializer.validated_data.get('preferences', {})
        preference_preset = serializer.validated_data.get('preference_preset')
        
        # Apply Preset Preference
        if preference_preset and preference_preset in PREFERENCE_PRESETS:
            from .recommender.preferences import filter_valid_preset_genres
            preset_preferences = filter_valid_preset_genres(preference_preset)
            preferences = {**preferences, **preset_preferences}
        
        try:
            # Get recommendations
            fast_recommendations = recommendation_engine.get_recommendations(
                input_tracks, n_recommendations, diversity_factor, preferences
            )
            
            if not fast_recommendations:
                error_msg = 'No recommendations found'
                if preferences:
                    error_msg += ' matching your preferences'
                return Response(
                    {'error': error_msg},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get full track details for recommendations
            recommended_track_ids = [rec['track_id'] for rec in fast_recommendations]
            recommended_tracks = Track.objects.filter(track_id__in=recommended_track_ids)
            track_map = {track.track_id: track for track in recommended_tracks}
            
            # Get input track details
            input_track_objects = []
            for track_id in input_tracks:
                try:
                    track = Track.objects.get(track_id=track_id)
                    input_track_objects.append(track)
                except Track.DoesNotExist:
                    pass
            
            if not input_track_objects:
                return Response(
                    {'error': 'Input track(s) not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Use first input track for explanation (or combine features for multiple)
            primary_input_track = input_track_objects[0]
            
            # Prepare recommendations in the new format
            recommendations = []
            for fast_rec in fast_recommendations:
                track = track_map.get(fast_rec['track_id'])
                if track:
                    # Generate explanation
                    explanation = generate_recommendation_explanation(
                        primary_input_track, track, fast_rec['similarity'], explanation_detail
                    )
                    
                    # Create recommendation data with serialized track data
                    recommendation_data = {
                        'recommended_track': TrackSerializer(track).data,  # Serialize the track
                        'similarity_score': fast_rec['similarity'],
                        'explanation': explanation
                    }
                    
                    recommendations.append(recommendation_data)
            
            # Prepare the response data with all input tracks together
            response_data = {
                'input_tracks': [TrackSerializer(track).data for track in input_track_objects],  # All input tracks together
                'recommendations': recommendations
            }
            
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {'error': f'Internal server error: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# CRUD for Tracks
class TrackInfoView(APIView):
    
    # Get Tracks
    def get(self, request, track_id=None):
        # Check if track_id exists
        if track_id:
            return self._get_track_detail(request, track_id)
        
        # Check if track_id is provided as a query parameter
        track_id_from_query = request.GET.get('track_id')
        if track_id_from_query:
            return self._get_track_detail(request, track_id_from_query)
        
        # Check if there's a search query
        search_query = request.GET.get('q', '').strip()
        if search_query:
            return self._get_track_list(request)
        
        # GetAllTracks: List tracks with pagination
        return self._get_track_list(request)
    
    # Create a new track
    def post(self, request):
        serializer = TrackDetailSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Check if track already exists
                track_id = serializer.validated_data.get('track_id')
                if Track.objects.filter(track_id=track_id).exists():
                    return Response(
                        {'error': f'Track with ID {track_id} already exists'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Create the track
                track = serializer.save()
                return Response(
                    TrackDetailSerializer(track).data,
                    status=status.HTTP_200_OK
                )
            except Exception as e:
                return Response(
                    {'error': f'Error creating track: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # Update an existing track
    def patch(self, request, track_id=None):
        # Check if track_id is provided as a URL path parameter
        if track_id:
            return self._patch_track(request, track_id)
        
        # Check if track_id is provided as a query parameter
        track_id_from_query = request.GET.get('track_id')
        if track_id_from_query:
            return self._patch_track(request, track_id_from_query)
        
        return Response(
            {'error': 'Track ID is required for update. Provide it in URL path or as query parameter'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Helper method to handle PATCH requests
    def _patch_track(self, request, track_id):
        try:
            track = Track.objects.get(track_id=track_id)
        except Track.DoesNotExist:
            raise Http404("Track not found")
        
        serializer = TrackDetailSerializer(track, data=request.data, partial=True)
        if serializer.is_valid():
            try:
                updated_track = serializer.save()
                return Response(TrackDetailSerializer(updated_track).data)
            except Exception as e:
                return Response(
                    {'error': f'Error updating track: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # Delete a track
    def delete(self, request, track_id=None):
        # Check if track_id is provided as a URL path parameter
        if track_id:
            return self._delete_track(track_id)
        
        # Check if track_id is provided as a query parameter
        track_id_from_query = request.GET.get('track_id')
        if track_id_from_query:
            return self._delete_track(track_id_from_query)
        
        return Response(
            {'error': 'Track ID is required for deletion. Provide it in URL path or as query parameter'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    def _delete_track(self, track_id):
        """Helper method to handle DELETE requests"""
        try:
            track = Track.objects.get(track_id=track_id)
            track.delete()
            return Response(
                {'message': f'Track {track_id} deleted successfully'},
                status=status.HTTP_200_OK
            )
        except Track.DoesNotExist:
            raise Http404("Track not found")
        except Exception as e:
            return Response(
                {'error': f'Error deleting track: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # Get details of a specific track
    def _get_track_detail(self, request, track_id):
        try:
            track = Track.objects.get(track_id=track_id)
            serializer = TrackDetailSerializer(track)
            return Response(serializer.data)
        except Track.DoesNotExist:
            raise Http404("Track not found")
    
    # Get list of tracks with optional search and pagination
    def _get_track_list(self, request):
        search_query = request.GET.get('q', '').strip()
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))

        if search_query:
            # Search tracks
            if len(search_query) < 2:
                return Response(
                    {'error': 'Search query must be at least 2 characters long'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            queryset = Track.objects.filter(
                Q(track_name__icontains=search_query) |
                Q(artists__icontains=search_query) |
                Q(album_name__icontains=search_query)
            )
        else:
            queryset = Track.objects.all()

        total_count = queryset.count()
        total_pages = (total_count + page_size - 1) // page_size  # ceil division

        start = (page - 1) * page_size
        end = start + page_size
        tracks = queryset[start:end]

        serializer = TrackDetailSerializer(tracks, many=True)
        return Response({
            'results': serializer.data,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'total_count': total_count
        })
