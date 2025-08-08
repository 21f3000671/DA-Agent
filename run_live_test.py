import requests
import os
import json

# Define the url and the file to be uploaded
url = "http://127.0.0.1:8000/api/"
file_path = "questions.txt"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    files = {
        'files': (os.path.basename(file_path), open(file_path, 'rb'), 'text/plain')
    }

    print("Sending request to the agent... This may take a moment.")

    try:
        # Send the POST request, with a generous timeout for the agent to work
        response = requests.post(url, files=files, timeout=300) # 5 minute timeout

        # Print the response from the server
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        # Pretty-print the JSON response
        try:
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print("Response is not valid JSON.")
            print(response.text)

    except requests.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")
        print("Please ensure the server is running.")
    except requests.exceptions.ReadTimeout as e:
        print(f"Request timed out: {e}")
        print("The agent took too long to respond.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the opened file handle
        files['files'][1].close()
