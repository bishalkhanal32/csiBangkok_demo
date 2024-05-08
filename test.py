import requests

# Define the API endpoint URL
url = "http://172.17.0.2:8000/predict"

# Prepare the JSON data
# data = {"data": [1., 0.18181819, 0.18181819, 0.2857143, 0.45901638, 0.85365856, 1., 1.]}
data = {"data": [9000., 4500., 4500., 30., 580., 1260., 30., 410.]}
# data = {"data": [4000., 4000., 8000., 10., 270., 480., 20., 380.]}
# Set the headers
headers = {"Content-Type": "application/json"}

# Send the POST request
response = requests.post(url, json=data, headers=headers)

# Check for successful response
if response.status_code == 200:
  # Print the response data
  print(response.json())
else:
  print(f"Error: API request failed with status code {response.status_code}")
