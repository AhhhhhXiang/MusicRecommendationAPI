from django.apps import AppConfig
import os
from django.conf import settings
from .recommender.engine import TrackRecommender

class NexttrackConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'nexttrack'

    def ready(self):
        csv_path = os.path.join(settings.BASE_DIR, 'nexttrack_project', 'data', 'tracks.csv')
        self.recommender = TrackRecommender(csv_path)