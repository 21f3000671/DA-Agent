import requests
import json

# Create test data
with open("specific_test.txt", "r") as f:
    questions_content = f.read()

with open("test_data.csv", "w") as f:
    f.write("price,sales,region\n10,100,North\n20,80,South\n15,90,North\n25,70,South\n30,60,East")

# Test with specific questions
url = "http://127.0.0.1:8000/api/"
files = [
    ('files', ('questions.txt', open('specific_test.txt', 'rb'), 'text/plain')),
    ('files', ('data.csv', open('test_data.csv', 'rb'), 'text/csv'))
]

try:
    print("Testing with specific questions...")
    print(f"Question: {questions_content}")
    
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nQuestions:")
        for q in result.get('questions', []):
            print(f"- {q}")
        
        print("\nAnswers:")
        for a in result.get('answers', []):
            print(f"- {a}")
            
        print(f"\nAnalysis Summary: {result.get('analysis_summary', 'N/A')}")
        print(f"Data Insights: {result.get('data_insights', [])}")
    else:
        print("Error:", response.text)

except Exception as e:
    print(f"Error: {e}")

finally:
    # Clean up
    for f_tuple in files:
        f_tuple[1][1].close()
    import os
    os.remove("test_data.csv")