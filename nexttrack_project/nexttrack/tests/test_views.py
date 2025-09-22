from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from nexttrack.models import Track

class TrackInfoViewTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.track = Track.objects.create(
            track_id="2plbrEY59IikOBgBGLjaoe",
            track_name="Die With A Smile",
            artists="Lady Gaga, Bruno Mars",
            album_name="Die With A Smile",
            popularity=100,
            duration_ms=251668,
            danceability=0.521,
            energy=0.592,
            key=6,
            loudness=-7.777,
            mode=0,
            speechiness=0.0304,
            acousticness=0.308,
            instrumentalness=0,
            liveness=0.122,
            valence=0.535,
            tempo=157.969,
            time_signature=3,
            track_genre="pop"
        )

    # Test GetTrack
    def test_get_track_detail(self):
        url = reverse("track_detail", args=[self.track.track_id])  
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["track_id"], "2plbrEY59IikOBgBGLjaoe")

    # Test CreateTrack
    def test_create_track(self):
        url = reverse("track_list")
        data = {
            "track_id": "new123",
            "track_name": "New Song",
            "artists": "New Artist",
            "album_name": "New Album",
            "popularity": 50,
            "duration_ms": 200000,
            "danceability": 0.4,
            "energy": 0.7,
            "key": 5,
            "loudness": -8.5,
            "mode": 1,
            "speechiness": 0.05,
            "acousticness": 0.2,
            "instrumentalness": 0.01,
            "liveness": 0.1,
            "valence": 0.6,
            "tempo": 110,
            "time_signature": 4,
            "track_genre": "rock"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Track.objects.count(), 2)

    # Test UpdateTrack
    def test_update_track(self):
        url = reverse("track_detail", args=[self.track.track_id])
        data = {"track_name": "Updated Song"}
        response = self.client.patch(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.track.refresh_from_db()
        self.assertEqual(self.track.track_name, "Updated Song")

    # Test DeleteTrack
    def test_delete_track(self):
        url = reverse("track_detail", args=[self.track.track_id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(Track.objects.filter(track_id="2plbrEY59IikOBgBGLjaoe").exists())


class NextTrackViewTests(APITestCase):
    def setUp(self):
        self.client = APIClient()

        self.track1 = Track.objects.create(
            track_id="4qPNDBW1i3p13qLCt0Ki3A",
            track_name="Dummy Song 1",
            artists="Artist 1",
            album_name="Album 1",
            popularity=70,
            duration_ms=210000,
            danceability=0.6,
            energy=0.7,
            key=5,
            loudness=-6.5,
            mode=1,
            speechiness=0.04,
            acousticness=0.2,
            instrumentalness=0.0,
            liveness=0.1,
            valence=0.5,
            tempo=120,
            time_signature=4,
            track_genre="pop"
        )

        self.track2 = Track.objects.create(
            track_id="1iJBSr7s7jYXzM8EGcbK5b",
            track_name="Dummy Song 2",
            artists="Artist 2",
            album_name="Album 2",
            popularity=65,
            duration_ms=200000,
            danceability=0.55,
            energy=0.65,
            key=7,
            loudness=-7.0,
            mode=0,
            speechiness=0.05,
            acousticness=0.25,
            instrumentalness=0.0,
            liveness=0.12,
            valence=0.55,
            tempo=115,
            time_signature=4,
            track_genre="rock"
        )

    # Test NextTrack
    def test_recommend_next_track_valid(self):
        url = reverse("next_track")
        data = {
            "track_ids": [
                self.track1.track_id,
                self.track2.track_id
            ],
            "n_recommendations": 5,
            "explanation_detail": True
        }
        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("recommendations", response.data)

    # Test NextTrack (non-existing track id)
    def test_recommend_next_track_invalid(self):
        url = reverse("next_track")
        data = {"track_id": "nonexistent123"}
        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
