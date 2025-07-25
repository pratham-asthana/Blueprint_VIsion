import requests
import numpy as np 

# Replace with your Render URL
url = "https://blueprint-vision.onrender.com/detect"

# Replace with path to your custom image
image_path = "test.jpg"

with open(image_path, "rb") as image_file:
    response = requests.post(url, files={"file": image_file})

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
