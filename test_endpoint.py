import requests
import os

# Create dummy files for testing
with open("questions.txt", "w") as f:
    f.write("This is a test question.")

with open("data.csv", "w") as f:
    f.write("col1,col2\n1,2")

# Define the url and the files to be uploaded
url = "http://127.0.0.1:8000/api/"
files = [
    ('files', ('questions.txt', open('questions.txt', 'rb'), 'text/plain')),
    ('files', ('data.csv', open('data.csv', 'rb'), 'text/csv'))
]

try:
    # Send the POST request
    response = requests.post(url, files=files)

    # Print the response from the server
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())

except requests.exceptions.ConnectionError as e:
    print(f"Connection failed: {e}")
    print("Please ensure the Uvicorn server is running.")

finally:
    # Clean up the dummy files
    for f_tuple in files:
        f_tuple[1][1].close()
    os.remove("questions.txt")
    os.remove("data.csv")
