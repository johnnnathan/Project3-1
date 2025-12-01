import requests

url = "http://127.0.0.1:5000/api/process_events"
payload = {
    "events": [{"x": 10, "y": 20, "p": 1}],
    "width": 320,
    "height": 240,
    "blur_faces": True
}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
