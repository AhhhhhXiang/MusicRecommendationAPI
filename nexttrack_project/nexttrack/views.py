from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.apps import apps
from .spotify_service import get_track_preview
import math

def _get_recommender():
    return apps.get_app_config('nexttrack').recommender

class NextTrackView(APIView):
    def post(self, request):
        track_id = request.data.get('track_id')

        if not track_id:
            return Response({'error': 'track_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        recommender = _get_recommender()
        if not recommender:
            return Response({'error': 'Recommender not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        results = recommender.recommend(track_id)
        if not results:
            return Response({'error': 'Track not found or no recommendations'}, status=status.HTTP_404_NOT_FOUND)

        # Enhance each track with Spotify data (name, artist, preview_url)
        enhanced_results = []
        for track in results:
            spotify_info = get_track_preview(track['track_id'])
            enhanced_results.append({
                "track_id": track['track_id'],
                "track_name": spotify_info.get("track_name", track.get("track_name")),
                "artists": spotify_info.get("artists", track.get("artists")),
                "preview_url": spotify_info.get("preview_url"),
                "similarity": track.get("similarity")
            })

        return Response({"RecommendedTrackResponse": enhanced_results}, status=status.HTTP_200_OK)

class TrackInfoView(APIView):
    def get(self, request, track_id=None):
        recommender = _get_recommender()
        if recommender is None:
            return Response({'error': 'Recommender not available.'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        df = recommender.df

        # --- Get single track info ---
        q_track_id = track_id or request.query_params.get('track_id')
        if q_track_id:
            row = df[df['track_id'] == q_track_id]
            if row.empty:
                return Response({'error': f'Track {q_track_id} not found.'},
                                status=status.HTTP_404_NOT_FOUND)

            # Filter only required fields
            record = row.iloc[0][[
                'track_id', 'track_name', 'artists', 'track_genre',
                'danceability', 'energy', 'key', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'tempo'
            ]].to_dict()

            return Response({"GetTrackResponse": record}, status=status.HTTP_200_OK)

        # --- Get All Tracks with pagination ---
        try:
            page = int(request.query_params.get('page', 1))
            page_size = int(request.query_params.get('page_size', 50))
        except ValueError:
            return Response({'error': 'page and page_size must be integers.'},
                            status=status.HTTP_400_BAD_REQUEST)

        page = max(1, page)

        total = len(df)
        total_pages = math.ceil(total / page_size)
        if page > total_pages and total_pages > 0:
            page = total_pages

        start = (page - 1) * page_size
        end = start + page_size

        # Filter required columns for all tracks
        page_df = df.iloc[start:end][[
            'track_id', 'track_name', 'artists', 'track_genre',
            'danceability', 'energy', 'key', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'tempo'
        ]]

        items = page_df.to_dict(orient='records')

        payload = {
            'count': total,
            'page': page,
            'total_pages': total_pages,
            'results': items,
        }
        return Response({"GetAllTrackResponse": payload}, status=status.HTTP_200_OK)