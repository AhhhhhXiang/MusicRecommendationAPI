from django.apps import AppConfig

class NexttrackConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'nexttrack'
    label = 'nexttrack'
    
    def ready(self):
        # Import and initialize the recommendation engine
        try:
            from .recommender.engine import recommendation_engine
            print("Initializing recommendation engine...")
            recommendation_engine.initialize()
        except Exception as e:
            print(f"Warning: Could not initialize recommendation engine: {str(e)}")