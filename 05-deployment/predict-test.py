import requests

url = "http://localhost:8080/predict"
# client = {"job": "unknown", "duration": 270, "poutcome": "failure"}

client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json=client).json()

# response = requests.post(url, json=client).json()
print(response)
