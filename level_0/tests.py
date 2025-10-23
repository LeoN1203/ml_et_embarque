import requests

data = {"size": 1000, "nb_rooms": 1, "garden": 1}
r = requests.post("http://4.210.225.87:8000/predict", json=data)
print(r.json())
