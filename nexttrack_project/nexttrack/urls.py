from django.urls import path
from .views import NextTrackView, TrackInfoView

urlpatterns = [
    path('api/v0/next-track/', NextTrackView.as_view(), name='next_track'),
    path('api/v0/tracks/', TrackInfoView.as_view(), name='track_list'),
    path('api/v0/tracks/<str:track_id>/', TrackInfoView.as_view(), name='track_detail'),
]