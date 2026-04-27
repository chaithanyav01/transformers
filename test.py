import requests

url = "http://18.61.84.33:8000/generate"
data = {"start": "Shubham", "max_new_tokens": 100}

response = requests.post(url, json=data)
res = response.json()

print(res["generated_text"])