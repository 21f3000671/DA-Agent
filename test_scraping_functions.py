#!/usr/bin/env python3
"""
Standalone test for the web scraping functions
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from agent.main import detect_urls, scrape_web_data, clean_scraped_dataframe

def test_url_detection():
    """Test URL detection function"""
    questions = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
"""
    
    print("Testing URL detection...")
    urls = detect_urls(questions)
    print(f"Detected URLs: {urls}")
    return urls

def test_web_scraping():
    """Test web scraping function"""
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    data_context = {}
    
    print(f"Testing web scraping for: {url}")
    result = scrape_web_data(url, data_context)
    
    print(f"Scraping success: {result['success']}")
    if result['success']:
        print(f"Scraped tables: {len(result['scraped_tables'])}")
        for table_info in result['scraped_tables']:
            print(f"- {table_info['name']}: {table_info['shape']} with columns {table_info['columns']}")
        
        # Show sample data from the first table
        if result['scraped_tables']:
            first_table_name = result['scraped_tables'][0]['name']
            df = result['data_context'][first_table_name]
            print(f"\nSample data from {first_table_name}:")
            print(df.head(3))
            print(f"Data types:\n{df.dtypes}")
    else:
        print(f"Scraping failed: {result['error']}")
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING WEB SCRAPING FUNCTIONS")
    print("=" * 60)
    
    # Test 1: URL Detection
    urls = test_url_detection()
    
    print("\n" + "-" * 40)
    
    # Test 2: Web Scraping
    if urls:
        result = test_web_scraping()
        
        if result['success']:
            print("\n✅ All tests passed! Web scraping functions work correctly.")
        else:
            print("\n❌ Web scraping failed.")
    else:
        print("\n❌ No URLs detected.")