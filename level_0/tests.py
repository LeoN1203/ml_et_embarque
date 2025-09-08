# import requests
# r = requests.get("http://127.0.0.1:8000/predict")
# print(r.json())  # {'y_pred': 2}

import requests
data = {
  "size": 1000,
  "nb_rooms": 1,
  "garden": 1
}
r = requests.post("http://4.210.225.87:5045/predict", json=data)
print(r.json())
