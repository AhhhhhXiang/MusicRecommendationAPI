from django.core.management.base import BaseCommand
from django.apps import apps

class Command(BaseCommand):
    help = 'Populate track embeddings from ProcessedTrack data'
    
    def handle(self, *args, **options):
        self.stdout.write('Starting embedding population...')
        
        # Import here to avoid circular imports
        from nexttrack.recommender.engine import recommendation_engine
        from nexttrack.models import ProcessedTrack, TrackEmbedding
        
        # Check if we have processed tracks
        processed_count = ProcessedTrack.objects.count()
        if processed_count == 0:
            self.stdout.write(
                self.style.ERROR('No ProcessedTrack records found. Please ensure your ProcessedTrack table is populated.')
            )
            return
        
        # Check if embeddings already exist
        embedding_count = TrackEmbedding.objects.count()
        if embedding_count > 0:
            self.stdout.write(
                self.style.WARNING(f'Found {embedding_count} existing embeddings. Deleting them...')
            )
            TrackEmbedding.objects.all().delete()
        
        # Initialize the engine and generate embeddings
        try:
            success = recommendation_engine.initialize()
            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully populated {len(recommendation_engine.track_ids) if hasattr(recommendation_engine, "track_ids") else 0} embeddings!'
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR('Failed to initialize recommendation engine')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error populating embeddings: {str(e)}')
            )
            import traceback
            traceback.print_exc()