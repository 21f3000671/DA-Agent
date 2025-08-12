import requests
import json

# Test the Wikipedia scraping functionality with the courts questions
def test_wikipedia_scraping():
    """Test the system with the Wikipedia URL that was causing errors."""
    
    # Create a test questions file with the Wikipedia URL
    questions_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
"""

    # API endpoint
    url = "http://127.0.0.1:8000/api/"
    
    # Prepare files - only sending questions.txt, no CSV files
    files = [
        ('files', ('questions.txt', questions_content.encode('utf-8'), 'text/plain'))
    ]
    
    try:
        print("Testing Wikipedia scraping with no CSV files provided...")
        print(f"Questions being sent:\n{'-'*50}")
        print(questions_content)
        print(f"{'-'*50}")
        
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS! Web scraping worked!")
            
            # Print the response structure
            print("\nResponse structure:")
            for key in result.keys():
                print(f"- {key}: {type(result[key])}")
            
            # Print questions and answers if available
            if 'questions' in result:
                print(f"\nQuestions processed: {len(result['questions'])}")
                for i, q in enumerate(result['questions'], 1):
                    print(f"{i}. {q}")
            
            if 'answers' in result:
                print(f"\nAnswers provided: {len(result['answers'])}")
                for i, a in enumerate(result['answers'], 1):
                    print(f"{i}. {a}")
            
            if 'analysis_summary' in result:
                print(f"\nAnalysis Summary: {result['analysis_summary']}")
                
            # Check if visualizations were created
            if 'visualizations' in result and result['visualizations']:
                print(f"\nVisualizations created: {len(result['visualizations'])}")
                
        else:
            print(f"❌ ERROR! Status Code: {response.status_code}")
            print("Response:", response.text)
            
            # Try to parse error details
            try:
                error_data = response.json()
                print("Error details:", error_data)
            except:
                print("Could not parse error response as JSON")
                
    except Exception as e:
        print(f"❌ EXCEPTION occurred: {e}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_wikipedia_scraping()