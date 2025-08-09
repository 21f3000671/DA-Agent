import openai
import json
import pandas as pd
import logging
import os
import io
import traceback
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assuming UploadFile is a type, for type hinting. In practice, we get the real object.
from fastapi import UploadFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_python_code(code: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Python code in a controlled environment with access to data context.
    Returns updated data context and any results.
    """
    # Create a safe execution environment with necessary imports
    try:
        import numpy as np
        import requests
        import base64
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        from bs4 import BeautifulSoup
        import duckdb
        from sklearn.linear_model import LinearRegression
        import networkx as nx
        
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
            'requests': requests,
            'json': json,
            'io': io,
            'base64': base64,
            'matplotlib': matplotlib,
            'plt': plt,
            'seaborn': sns,
            'sns': sns,
            'BeautifulSoup': BeautifulSoup,
            'duckdb': duckdb,
            'LinearRegression': LinearRegression,
            'nx': nx,
            'networkx': nx,
            'data_context': data_context,
            'result': None
        }
        
        # Set matplotlib to non-interactive backend
        matplotlib.use('Agg')
        
        # Execute the code
        exec(code, exec_globals)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        result = exec_globals.get('result')
        if result is not None:
            result = convert_numpy_types(result)
        
        # Return updated data context and any result
        return {
            'success': True,
            'data_context': exec_globals.get('data_context', data_context),
            'result': result,
            'error': None
        }
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        logger.error(f"Code that failed:\n{code}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return {
            'success': False,
            'data_context': data_context,
            'result': None,
            'error': str(e)
        }

# Get LLM configuration from environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)  # For custom endpoints
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")  # Fallback to OPENAI_API_KEY

# Log configuration for debugging
logger.info(f"LLM Configuration:")
logger.info(f"  Provider: {LLM_PROVIDER}")
logger.info(f"  Model: {LLM_MODEL}")
logger.info(f"  Base URL: {LLM_BASE_URL}")
logger.info(f"  API Key present: {'Yes' if LLM_API_KEY else 'No'}")

# Initialize the LLM client based on provider
client = None
# List of providers that use OpenAI-compatible API
openai_compatible_providers = ["openai", "openrouter", "together", "deepseek", "groq", "anthropic", "custom"]

if LLM_PROVIDER.lower() in openai_compatible_providers:
    try:
        if not LLM_API_KEY:
            logger.error(f"LLM_API_KEY or OPENAI_API_KEY is required for {LLM_PROVIDER} provider")
        else:
            # Build client parameters
            client_params = {"api_key": LLM_API_KEY}
            if LLM_BASE_URL:
                client_params["base_url"] = LLM_BASE_URL
            elif LLM_PROVIDER.lower() == "openrouter":
                # Default OpenRouter base URL if not provided
                client_params["base_url"] = "https://openrouter.ai/api/v1"
            
            client = openai.OpenAI(**client_params)
            logger.info(f"Successfully initialized {LLM_PROVIDER} client with model: {LLM_MODEL}")
            logger.info(f"Using base URL: {client_params.get('base_url', 'https://api.openai.com/v1')}")
    except Exception as e:
        logger.error(f"Failed to initialize {LLM_PROVIDER} client: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
elif LLM_PROVIDER.lower() == "google" or LLM_PROVIDER.lower() == "gemini":
    try:
        if not LLM_API_KEY:
            logger.error("LLM_API_KEY is required for Google Gemini provider")
        else:
            # Use Google AI Studio API with OpenAI-compatible interface via LiteLLM or direct API
            # Default to Google AI Studio endpoint
            base_url = LLM_BASE_URL or "https://generativelanguage.googleapis.com/v1beta"
            
            # For now, recommend using Gemini via OpenRouter for better compatibility
            if not LLM_BASE_URL:
                logger.warning("Direct Google Gemini integration is complex. Consider using:")
                logger.warning("LLM_PROVIDER=openrouter")
                logger.warning("LLM_MODEL=google/gemini-flash-1.5")
                logger.warning("This provides better tool calling support.")
                
                # Use a simple wrapper that explains the limitation
                class GeminiDirectWrapper:
                    def __init__(self):
                        pass
                    
                    def chat_completions_create(self, **kwargs):
                        raise NotImplementedError(
                            "Direct Google Gemini integration requires additional setup. "
                            "Please use Gemini via OpenRouter: LLM_PROVIDER=openrouter, LLM_MODEL=google/gemini-flash-1.5"
                        )
                
                client = GeminiDirectWrapper()
            else:
                # If user provides a base URL, try to use it with OpenAI interface
                client = openai.OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=base_url
                )
                logger.info(f"Initialized Google Gemini client with custom base URL: {base_url}")
    except Exception as e:
        logger.error(f"Failed to initialize Google Gemini client: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
else:
    logger.error(f"Unsupported LLM provider: {LLM_PROVIDER}")
    logger.error(f"Supported providers: {', '.join(openai_compatible_providers + ['google', 'gemini'])}")

if not client:
    logger.error("No LLM client initialized. Please check your configuration.")
    logger.error("Make sure you have set the following environment variables:")
    logger.error("  LLM_PROVIDER (e.g., 'openai')")
    logger.error("  LLM_MODEL (e.g., 'gpt-4o')")
    logger.error("  LLM_API_KEY or OPENAI_API_KEY")
    logger.error("  LLM_BASE_URL (for custom endpoints)")

# Three-stage prompt system
DATA_GATHERING_PROMPT = """
You are a Data Gathering Specialist. Your task is to write Python code to collect and prepare data based on the user's requirements.

Available imports and libraries:
- pandas as pd
- requests
- BeautifulSoup from bs4
- duckdb
- json
- io

Your capabilities:
1. **Web Scraping**: Use requests + BeautifulSoup or pandas.read_html() to scrape tables from websites
2. **File Processing**: Read CSV, JSON, and other data files using pandas
3. **Database Queries**: Use DuckDB for SQL queries on large datasets or remote parquet files
4. **Data Loading**: Process uploaded files and create DataFrames

Code Templates:

# Web scraping example:
# import requests
# import pandas as pd
# tables = pd.read_html(url)
# data_context['scraped_data'] = tables[0]

# DuckDB query example:
# import duckdb
# conn = duckdb.connect()
# result = conn.execute("SELECT * FROM 'data.csv' LIMIT 10").fetchdf()
# data_context['query_result'] = result

# File processing example:
# df = pd.read_csv('filename.csv')
# data_context['processed_data'] = df

IMPORTANT: 
- Store all DataFrames in the 'data_context' dictionary with descriptive names
- Use safe type conversions: pd.to_numeric(col, errors='coerce')
- Handle errors gracefully
- Set 'result' variable to a summary of what data was gathered
- Your code should be complete and executable

Write Python code to gather the required data. End your response with the code block only.
"""

DATA_ANALYSIS_PROMPT = """
You are a Data Analysis Specialist. Your task is to write Python code to perform statistical analysis and calculations on the available data.

Available imports and libraries:
- pandas as pd
- numpy as np
- scipy.stats for statistical functions
- sklearn.linear_model.LinearRegression for regression
- duckdb for complex SQL operations
- networkx as nx for network analysis

Your capabilities:
1. **Statistical Analysis**: Correlations, means, medians, standard deviations
2. **Regression Analysis**: Linear regression, polynomial fitting
3. **Data Transformations**: Grouping, aggregations, pivoting
4. **Advanced Calculations**: Custom formulas and mathematical operations

Code Templates:

# Correlation example:
# df = data_context['dataset_name']
# df['col1'] = pd.to_numeric(df['col1'], errors='coerce')
# df['col2'] = pd.to_numeric(df['col2'], errors='coerce')
# correlation = df['col1'].corr(df['col2'])
# result['correlation'] = correlation

# Linear regression example:
# from sklearn.linear_model import LinearRegression
# X = df[['x_column']].dropna()
# y = df['y_column'].dropna()
# model = LinearRegression().fit(X, y)
# result['slope'] = model.coef_[0]
# result['intercept'] = model.intercept_

# Statistical calculations:
# result['mean'] = df['column'].mean()
# result['median'] = df['column'].median()
# result['sum'] = df['column'].sum()

IMPORTANT:
- Access data using data_context['dataset_name']
- Store analysis results in a 'result' dictionary
- Handle missing values and data type conversions
- Use error handling for robust calculations
- Your code should be complete and executable

Write Python code to perform the required analysis. End your response with the code block only.
"""

PRESENTATION_PROMPT = """
You are a Data Presentation Specialist. Your task is to write Python code to create visualizations and format the final output as JSON.

Available imports and libraries:
- matplotlib.pyplot as plt (set to Agg backend)
- seaborn as sns
- pandas as pd
- base64 for encoding images
- json for output formatting
- io for handling image buffers

Your capabilities:
1. **Visualizations**: Scatter plots, bar charts, line plots, histograms
2. **Plot Customization**: Colors, labels, titles, legends, styling
3. **Image Encoding**: Convert plots to base64 data URI strings
4. **JSON Formatting**: Structure final results according to requirements

Code Templates:

# Basic plot with base64 encoding:
# plt.figure(figsize=(10, 6))
# plt.plot(data, color='red', linestyle='-')
# plt.title('Chart Title')
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# 
# buffer = io.BytesIO()
# plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
# plt.close()
# buffer.seek(0)
# img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
# result['chart'] = f"data:image/png;base64,{img_base64}"

# Bar chart example:
# plt.figure(figsize=(8, 6))
# plt.bar(categories, values, color='blue')
# plt.title('Title')

# Scatter plot with regression line:
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, alpha=0.6)
# plt.plot(x, regression_line, color='red', linestyle='--')

IMPORTANT:
- Keep image sizes under 100kB by using reasonable figure sizes and DPI
- Always use plt.close() after saving plots to free memory
- Encode images as base64 data URI: f"data:image/png;base64,{base64_string}"
- Structure final output as valid JSON matching the required format
- Set 'result' variable to the final JSON structure
- Your code should be complete and executable

Write Python code to create visualizations and format the final JSON output. End your response with the code block only.
"""

# The main three-stage agent orchestrator function
async def run_agent(questions: str, files: List[UploadFile]) -> Any:
    if not client:
        error_msg = (
            "LLM client is not initialized. Please check your environment variables:\n"
            f"LLM_PROVIDER: {LLM_PROVIDER}\n"
            f"LLM_MODEL: {LLM_MODEL}\n"
            f"LLM_BASE_URL: {LLM_BASE_URL}\n"
            f"API Key present: {'Yes' if LLM_API_KEY else 'No'}\n"
            "Make sure LLM_API_KEY or OPENAI_API_KEY is set correctly."
        )
        logger.error(error_msg)
        return {"error": error_msg}

    logger.info("Starting three-stage data agent...")

    # This dictionary will hold our data context
    data_context: Dict[str, Any] = {}

    # Read uploaded files into pandas DataFrames if they are CSVs
    file_info = []
    for file in files:
        filename = file.filename
        file_info.append(filename)
        if filename.endswith('.csv'):
            try:
                content = await file.read()
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                data_context[filename] = df
                logger.info(f"Loaded uploaded CSV '{filename}' with shape {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
            except Exception as e:
                logger.error(f"Failed to read CSV file {filename}: {e}")
    
    logger.info(f"Initial data_context keys: {list(data_context.keys())}")
    
    # Create data context summary for prompts
    data_summary = "\n".join([
        f"- {name}: {df.shape if isinstance(df, pd.DataFrame) else type(df).__name__}"
        for name, df in data_context.items()
    ])
    
    try:
        # STAGE 1: DATA GATHERING
        logger.info("=== STAGE 1: DATA GATHERING ===")
        
        # Skip data gathering if we already have sufficient data loaded
        needs_additional_data = (
            not data_context or 
            any(keyword in questions.lower() for keyword in ['scrape', 'fetch', 'url', 'web', 'http'])
        )
        
        if needs_additional_data:
            gathering_messages = [
                {"role": "system", "content": DATA_GATHERING_PROMPT},
                {
                    "role": "user", 
                    "content": f"Questions:\n{questions}\n\nFiles provided: {', '.join(file_info)}\n\nExisting data context:\n{data_summary}\n\nWrite Python code to gather any additional data needed. NOTE: Uploaded files are already available in data_context."
                }
            ]
            
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=gathering_messages
            )
            
            gathering_code = extract_code_from_response(response.choices[0].message.content)
            if gathering_code:
                logger.info("Executing data gathering code...")
                exec_result = execute_python_code(gathering_code, data_context)
                if exec_result['success']:
                    data_context = exec_result['data_context']
                    logger.info(f"Data gathering completed. New data_context keys: {list(data_context.keys())}")
                else:
                    logger.error(f"Data gathering failed: {exec_result['error']}")
                    return {"error": f"Data gathering stage failed: {exec_result['error']}"}
        else:
            logger.info("Skipping data gathering - sufficient data already available")
        
        # STAGE 2: DATA ANALYSIS
        logger.info("=== STAGE 2: DATA ANALYSIS ===")
        
        # Create detailed data information for analysis stage
        data_info = []
        for name, df in data_context.items():
            if isinstance(df, pd.DataFrame):
                data_info.append(f"- {name}: shape {df.shape}")
                data_info.append(f"  Columns: {list(df.columns)}")
                data_info.append(f"  Sample data:\n{df.head(3).to_string()}")
            else:
                data_info.append(f"- {name}: {type(df).__name__}")
        
        analysis_messages = [
            {"role": "system", "content": DATA_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": f"Questions:\n{questions}\n\nAvailable data:\n" + 
                          "\n".join(data_info) +
                          "\n\nWrite Python code to analyze the data and answer the questions. Use the EXACT column names shown above."
            }
        ]
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=analysis_messages
        )
        
        analysis_code = extract_code_from_response(response.choices[0].message.content)
        analysis_results = {}
        if analysis_code:
            logger.info("Executing data analysis code...")
            exec_result = execute_python_code(analysis_code, data_context)
            if exec_result['success']:
                data_context = exec_result['data_context']
                analysis_results = exec_result['result'] or {}
                logger.info(f"Data analysis completed. Results: {analysis_results}")
            else:
                logger.error(f"Data analysis failed: {exec_result['error']}")
                return {"error": f"Data analysis stage failed: {exec_result['error']}"}
        
        # STAGE 3: PRESENTATION
        logger.info("=== STAGE 3: PRESENTATION ===")
        
        # Create detailed data information for presentation stage
        data_info_detailed = []
        for name, df in data_context.items():
            if isinstance(df, pd.DataFrame):
                data_info_detailed.append(f"- {name}: shape {df.shape}")
                data_info_detailed.append(f"  Columns: {list(df.columns)}")
                data_info_detailed.append(f"  Sample data:\n{df.head(3).to_string()}")
            else:
                data_info_detailed.append(f"- {name}: {type(df).__name__}")
        
        presentation_messages = [
            {"role": "system", "content": PRESENTATION_PROMPT},
            {
                "role": "user",
                "content": f"Questions:\n{questions}\n\nAnalysis results:\n{json.dumps(analysis_results, default=str)}\n\nAvailable data:\n" +
                          "\n".join(data_info_detailed) +
                          "\n\nIMPORTANT: \n- Use data_context['dataset_name'] to access data - NEVER read from files!\n- Store your final result in the 'result' variable as a Python dictionary\n- Do NOT call json.dumps() - just create the dictionary\n- Convert numpy types to Python types (use int() and float())\n\nWrite Python code to create visualizations and format the final result dictionary."
            }
        ]
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=presentation_messages
        )
        
        presentation_code = extract_code_from_response(response.choices[0].message.content)
        if presentation_code:
            logger.info("Executing presentation code...")
            exec_result = execute_python_code(presentation_code, data_context)
            if exec_result['success']:
                final_result = exec_result['result']
                logger.info("Presentation completed successfully.")
                return final_result
            else:
                logger.error(f"Presentation failed: {exec_result['error']}")
                return {"error": f"Presentation stage failed: {exec_result['error']}"}
        
        # If no code was generated, return analysis results
        return analysis_results or {"error": "No results generated"}
        
    except Exception as e:
        logger.error(f"Error in three-stage agent: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Three-stage agent failed: {str(e)}"}

def extract_code_from_response(content: str) -> str:
    """
    Extract Python code from LLM response.
    """
    import re
    
    # Look for code blocks
    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code blocks, look for code after certain patterns
    lines = content.split('\n')
    code_started = False
    code_lines = []
    
    for line in lines:
        # Start collecting code after certain indicators
        if any(indicator in line.lower() for indicator in ['import ', 'def ', 'data_context', 'result =']):
            code_started = True
        
        if code_started:
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip() if code_lines else ""
