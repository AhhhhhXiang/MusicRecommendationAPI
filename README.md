# NextTrack

NextTrack is a RESTful music recommendation API built with Django.  
It suggests the next track based on a sequence of track IDs or user-defined preferences, using PostgreSQL for metadata storage and embeddings for recommendations.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Data Processing

Before running the server, you’ll need to process track metadata and generate embeddings.

### Load Tracks into Database

```bash
python manage.py process_tracks
```

### Generate Track Embeddings

```bash
python manage.py populate_embeddings
```

## Run the Server

Start the API locally:

```bash
python manage.py runserver
```