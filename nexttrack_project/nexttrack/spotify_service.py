import requests
import base64
from django.conf import settings

CLIENT_ID = "d5631395d58548c9bd4dfc1f35a71336"
CLIENT_SECRET = "55b0080c481141cc9904f32e099674f0"

def get_spotify_token():
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()
    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    return response.json().get("access_token")

def get_track_preview(track_id):
    token = get_spotify_token()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return {}
    
    data = response.json()
    return {
        "track_id": track_id,
        "track_name": data.get("name"),
        "artists": ", ".join(artist["name"] for artist in data.get("artists", [])),
        "preview_url": data.get("preview_url")
    }
