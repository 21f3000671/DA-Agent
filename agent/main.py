import openai
import json
import pandas as pd
import logging
import os
import io
import traceback
import re
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Assuming UploadFile is a type, for type hinting. In practice, we get the real object.
from fastapi import UploadFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_urls(text: str) -> List[str]:
    """
    Detect URLs in the given text using regex patterns.
    Returns a list of detected URLs including S3 URLs.
    """
    # Pattern for HTTP/HTTPS URLs
    http_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Pattern for S3 URLs (s3://bucket/key) - improved to handle more characters but stop at line breaks
    s3_pattern = re.compile(
        r's3://[a-zA-Z0-9\-\.]+/[a-zA-Z0-9\-\./\?=&_%+]+(?=\s|$)'
    )
    
    http_urls = http_pattern.findall(text)
    s3_urls = s3_pattern.findall(text)
    
    all_urls = http_urls + s3_urls
    logger.info(f"Detected URLs: {all_urls} (HTTP: {len(http_urls)}, S3: {len(s3_urls)})")
    return all_urls

def is_valid_url(url: str) -> bool:
    """Check if URL is valid and accessible."""
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def scrape_web_data(url: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for web scraping that the LLM can call.
    Attempts to scrape tables from the given URL and stores them in data_context.
    """
    try:
        # Basic URL validation
        if not is_valid_url(url):
            logger.info(f"Invalid URL format: {url}")
            return {
                'success': False,
                'data_context': data_context,
                'scraped_tables': [],
                'error': f"Invalid URL format: {url}"
            }
        
        logger.info(f"Attempting to scrape data from: {url}")
        
        # Try multiple methods to get tables
        scraped_data = []
        
        # Method 1: pandas.read_html (most common for tables)
        try:
            tables = pd.read_html(url)
            for i, table in enumerate(tables):
                # Clean and process the table
                table = clean_scraped_dataframe(table)
                table_name = f"scraped_table_{i+1}"
                data_context[table_name] = table
                scraped_data.append({
                    'name': table_name,
                    'shape': table.shape,
                    'columns': list(table.columns)
                })
                logger.info(f"Scraped table {i+1} with shape {table.shape}")
        except Exception as e:
            # Handle different types of errors with appropriate log levels
            error_msg = str(e).lower()
            if '404' in error_msg or 'not found' in error_msg:
                logger.info(f"URL not accessible (404): {url}")
            elif '403' in error_msg or 'forbidden' in error_msg:
                logger.info(f"URL access forbidden (403): {url}")
            elif 'timeout' in error_msg or 'timed out' in error_msg:
                logger.info(f"URL request timeout: {url}")
            elif 'no tables found' in error_msg:
                logger.info(f"No HTML tables found at: {url}")
            else:
                logger.warning(f"pandas.read_html failed for {url}: {e}")
        
        # Method 2: BeautifulSoup for more complex scraping if pandas failed
        if not scraped_data:
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP errors
                soup = BeautifulSoup(response.content, 'html.parser')
                tables = soup.find_all('table')
                
                for i, table in enumerate(tables):
                    df = parse_html_table_to_dataframe(table)
                    if df is not None and not df.empty:
                        df = clean_scraped_dataframe(df)
                        table_name = f"scraped_table_{i+1}"
                        data_context[table_name] = df
                        scraped_data.append({
                            'name': table_name,
                            'shape': df.shape,
                            'columns': list(df.columns)
                        })
                        logger.info(f"Scraped table {i+1} with BeautifulSoup, shape {df.shape}")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.info(f"URL not found (404): {url}")
                elif e.response.status_code == 403:
                    logger.info(f"URL access forbidden (403): {url}")
                else:
                    logger.warning(f"HTTP error {e.response.status_code}: {url}")
            except requests.exceptions.Timeout:
                logger.info(f"URL request timeout: {url}")
            except requests.exceptions.ConnectionError:
                logger.info(f"Connection error for URL: {url}")
            except Exception as e:
                logger.warning(f"BeautifulSoup scraping failed for {url}: {e}")
        
        if scraped_data:
            return {
                'success': True,
                'data_context': data_context,
                'scraped_tables': scraped_data,
                'error': None
            }
        else:
            logger.info(f"No tables found at URL: {url}")
            return {
                'success': False,
                'data_context': data_context,
                'scraped_tables': [],
                'error': f"No tables found at {url}. The page may not contain HTML tables or may be inaccessible."
            }
            
    except Exception as e:
        logger.error(f"Web scraping failed for {url}: {e}")
        return {
            'success': False,
            'data_context': data_context,
            'scraped_tables': [],
            'error': str(e)
        }

def clean_scraped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process scraped dataframes to handle data type issues.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Handle unnamed columns
    df.columns = [f"col_{i}" if str(col).startswith('Unnamed:') else str(col) for i, col in enumerate(df.columns)]
    
    # Convert all columns to string first to handle mixed types
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # Try to convert numeric columns safely
    for col in df.columns:
        try:
            # Try numeric conversion, but keep as string if it fails
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            # Only convert if most values are numeric (>50%) and not all are NaN
            valid_numerics = numeric_col.notna().sum()
            if valid_numerics > 0 and valid_numerics > len(df) * 0.5:
                df[col] = numeric_col
        except Exception as e:
            logger.warning(f"Could not process column {col} for numeric conversion: {e}")
            # Keep as string if any error occurs
            continue
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.loc[:, df.notna().any()]
    
    # Handle any remaining issues with string operations
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Ensure all values in object columns are strings
                df[col] = df[col].fillna('').astype(str)
            except Exception as e:
                logger.warning(f"Could not clean column {col}: {e}")
                continue
    
    return df

def parse_html_table_to_dataframe(table) -> pd.DataFrame:
    """
    Parse HTML table element to DataFrame using BeautifulSoup.
    """
    try:
        rows = []
        header_row = table.find('tr')
        
        # Extract headers
        headers = []
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            row_data = []
            for cell in row.find_all(['td', 'th']):
                row_data.append(cell.get_text(strip=True))
            if row_data:  # Only add non-empty rows
                rows.append(row_data)
        
        if rows and headers:
            # Ensure all rows have the same number of columns
            max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
            headers = headers + [f"col_{i}" for i in range(len(headers), max_cols)]
            
            # Pad rows to match header length
            padded_rows = []
            for row in rows:
                padded_row = row + [''] * (max_cols - len(row))
                padded_rows.append(padded_row[:max_cols])  # Trim if too long
            
            return pd.DataFrame(padded_rows, columns=headers[:max_cols])
        
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to parse HTML table: {e}")
        return pd.DataFrame()

def safe_numeric_convert(df, columns):
    """
    Safely convert specified columns to numeric, handling errors gracefully.
    
    Args:
        df: DataFrame to modify
        columns: List of column names or single column name to convert
        
    Returns:
        DataFrame with converted columns
    """
    if isinstance(columns, str):
        columns = [columns]
    
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def safe_plot_to_base64():
    """
    Safely convert current matplotlib plot to base64 string using PNG format.
    
    Returns:
        str: Base64 encoded PNG image as data URI
    """
    try:
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        # Create buffer and save plot
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode to base64
        plot_data = base64.b64encode(buffer.read()).decode()
        buffer.close()
        plt.close()
        
        # Return as data URI
        return f"data:image/png;base64,{plot_data}"
    
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        return f"Error creating visualization: {str(e)}"

def is_s3_url(url: str) -> bool:
    """Check if a URL is an S3 URL."""
    return url.startswith('s3://') or 's3.amazonaws.com' in url or '.s3.' in url

def download_from_s3(url: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download data from S3 and load into data_context.
    Supports s3:// URLs and HTTPS S3 URLs.
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        import tempfile
        import os
        from urllib.parse import urlparse
        
        logger.info(f"Downloading from S3: {url}")
        
        # Parse S3 URL
        if url.startswith('s3://'):
            # s3://bucket/key format
            parsed = urlparse(url)
            bucket_name = parsed.netloc
            key = parsed.path.lstrip('/')
        elif 's3.amazonaws.com' in url or '.s3.' in url:
            # HTTPS S3 URL format
            parsed = urlparse(url)
            if 's3.amazonaws.com' in parsed.netloc:
                # https://bucket.s3.amazonaws.com/key
                bucket_name = parsed.netloc.split('.s3.amazonaws.com')[0]
                key = parsed.path.lstrip('/')
            else:
                # https://s3.region.amazonaws.com/bucket/key
                path_parts = parsed.path.lstrip('/').split('/', 1)
                bucket_name = path_parts[0]
                key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            return {"success": False, "error": f"Unsupported S3 URL format: {url}"}
        
        # Create S3 client (will use default credentials or IAM role)
        try:
            s3_client = boto3.client('s3')
            # Test access with a head_object call
            s3_client.head_object(Bucket=bucket_name, Key=key)
        except NoCredentialsError:
            logger.warning("No AWS credentials found, attempting anonymous access")
            s3_client = boto3.client('s3', 
                                   config=boto3.session.Config(signature_version='UNSIGNED'))
        except ClientError as e:
            if e.response['Error']['Code'] == '403':
                logger.warning("Access denied, attempting anonymous access")
                s3_client = boto3.client('s3', 
                                       config=boto3.session.Config(signature_version='UNSIGNED'))
            else:
                return {"success": False, "error": f"S3 access error: {str(e)}"}
        
        # Determine file type from key
        file_extension = key.split('.')[-1].lower() if '.' in key else ''
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file_path = tmp_file.name
            try:
                # Download file
                s3_client.download_fileobj(bucket_name, key, tmp_file)
                
                logger.info(f"Downloaded {key} from bucket {bucket_name} to {tmp_file_path}")
                
                # Load data based on file type
                filename = os.path.basename(key)
                dataframe_key = f"s3_{filename}"
                
                if file_extension in ['csv']:
                    df = pd.read_csv(tmp_file_path)
                    data_context[dataframe_key] = df
                    logger.info(f"Loaded CSV with shape {df.shape} as '{dataframe_key}'")
                elif file_extension in ['json']:
                    df = pd.read_json(tmp_file_path)
                    data_context[dataframe_key] = df
                    logger.info(f"Loaded JSON with shape {df.shape} as '{dataframe_key}'")
                elif file_extension in ['parquet']:
                    df = pd.read_parquet(tmp_file_path)
                    data_context[dataframe_key] = df
                    logger.info(f"Loaded Parquet with shape {df.shape} as '{dataframe_key}'")
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(tmp_file_path)
                    data_context[dataframe_key] = df
                    logger.info(f"Loaded Excel with shape {df.shape} as '{dataframe_key}'")
                else:
                    # Try to load as text
                    with open(tmp_file_path, 'r') as f:
                        content = f.read()
                        data_context[f"s3_{filename}_content"] = content
                        logger.info(f"Loaded text file content as 's3_{filename}_content'")
                
                return {
                    "success": True, 
                    "data_context": data_context,
                    "message": f"Successfully downloaded and loaded {filename} from S3",
                    "filename": filename,
                    "key": dataframe_key
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
    except ImportError:
        return {"success": False, "error": "boto3 not available for S3 operations"}
    except Exception as e:
        logger.error(f"S3 download failed: {str(e)}")
        return {"success": False, "error": f"S3 download failed: {str(e)}"}

def execute_duckdb_query(query: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a DuckDB query with access to data_context DataFrames.
    """
    try:
        import duckdb
        
        logger.info(f"Executing DuckDB query: {query}")
        
        # Create DuckDB connection
        conn = duckdb.connect()
        
        # Register all DataFrames as virtual tables
        registered_tables = []
        for key, value in data_context.items():
            if isinstance(value, pd.DataFrame):
                # Clean table name (remove special characters)
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', key)
                conn.register(table_name, value)
                registered_tables.append((key, table_name))
                logger.info(f"Registered DataFrame '{key}' as table '{table_name}'")
        
        # Execute query
        result_df = conn.execute(query).fetchdf()
        
        # Store result in data_context
        result_key = "duckdb_query_result"
        counter = 1
        while result_key in data_context:
            result_key = f"duckdb_query_result_{counter}"
            counter += 1
            
        data_context[result_key] = result_df
        
        return {
            "success": True,
            "data_context": data_context,
            "result_key": result_key,
            "result_shape": result_df.shape,
            "registered_tables": registered_tables,
            "message": f"Query executed successfully, result stored as '{result_key}'"
        }
        
    except ImportError:
        return {"success": False, "error": "DuckDB not available"}
    except Exception as e:
        logger.error(f"DuckDB query failed: {str(e)}")
        return {"success": False, "error": f"DuckDB query failed: {str(e)}"}

def download_data_from_url(url: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Universal data download function that handles both S3 and HTTP URLs.
    """
    if is_s3_url(url):
        return download_from_s3(url, data_context)
    else:
        return scrape_web_data(url, data_context)

def fix_type_conversion_errors(code: str) -> str:
    """
    Automatically fix common type conversion errors in generated code.
    
    Args:
        code: Python code that failed with type conversion errors
        
    Returns:
        Fixed code with safe type conversions
    """
    import re
    
    fixed_code = code
    
    # Pattern 1: Replace .astype(float) with pd.to_numeric(..., errors='coerce')
    pattern1 = r"\.astype\(float\)"
    replacement1 = ""
    matches1 = re.finditer(pattern1, fixed_code)
    
    for match in reversed(list(matches1)):  # Reverse to maintain positions
        start, end = match.span()
        # Find the start of the expression (usually after = or [)
        expr_start = fixed_code.rfind('=', 0, start) + 1
        if expr_start == 0:
            expr_start = fixed_code.rfind('[', 0, start) + 1
        
        # Extract the column expression
        expr = fixed_code[expr_start:start].strip()
        replacement = f" pd.to_numeric({expr}, errors='coerce')"
        fixed_code = fixed_code[:expr_start] + replacement + fixed_code[end:]
    
    # Pattern 2: Replace .str.replace().astype(float) patterns
    pattern2 = r"(\w+\[.*?\]\.str\.replace\([^)]+\))\.astype\(float\)"
    replacement2 = r"pd.to_numeric(\1, errors='coerce')"
    fixed_code = re.sub(pattern2, replacement2, fixed_code)
    
    # Pattern 3: Replace pd.to_numeric without errors parameter
    pattern3 = r"pd\.to_numeric\(([^,)]+)\)"
    replacement3 = r"pd.to_numeric(\1, errors='coerce')"
    fixed_code = re.sub(pattern3, replacement3, fixed_code)
    
    return fixed_code

def execute_python_code(code: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Python code in a controlled environment with access to data context.
    Returns updated data context and any results.
    """
    # Create a safe execution environment with necessary imports
    try:
        import requests
        import base64
        from bs4 import BeautifulSoup
        
        # Optional heavy dependencies - fail gracefully if not available
        try:
            import numpy as np
        except ImportError:
            np = None
            
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use('Agg')  # Set to non-interactive backend
        except ImportError:
            matplotlib = None
            plt = None
            
        try:
            import seaborn as sns
        except ImportError:
            sns = None
            
        try:
            import duckdb
        except ImportError:
            duckdb = None
            
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            LinearRegression = None
            
        try:
            import networkx as nx
        except ImportError:
            nx = None
        
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': pd,
            'pandas': pd,
            'requests': requests,
            'json': json,
            'io': io,
            'base64': base64,
            'BeautifulSoup': BeautifulSoup,
            're': re,
            'data_context': data_context,
            'result': None,
            # Add questions variable if available
            'questions': data_context.get('questions', ''),
            # Helper functions for web scraping
            'detect_urls': detect_urls,
            'is_valid_url': is_valid_url,
            'scrape_web_data': scrape_web_data,
            'clean_scraped_dataframe': clean_scraped_dataframe,
            'parse_html_table_to_dataframe': parse_html_table_to_dataframe,
            # Helper function for safe type conversion
            'safe_numeric_convert': safe_numeric_convert,
            # Helper function for safe plotting
            'safe_plot_to_base64': safe_plot_to_base64,
            # S3 and data download functions
            'is_s3_url': is_s3_url,
            'download_from_s3': download_from_s3,
            'download_data_from_url': download_data_from_url,
            # DuckDB functions
            'execute_duckdb_query': execute_duckdb_query
        }
        
        # Add optional dependencies only if available
        if np is not None:
            exec_globals.update({'np': np, 'numpy': np})
        if matplotlib is not None and plt is not None:
            exec_globals.update({'matplotlib': matplotlib, 'plt': plt})
        if sns is not None:
            exec_globals.update({'seaborn': sns, 'sns': sns})
        if duckdb is not None:
            exec_globals['duckdb'] = duckdb
        if LinearRegression is not None:
            exec_globals['LinearRegression'] = LinearRegression
        if nx is not None:
            exec_globals.update({'nx': nx, 'networkx': nx})
        
        # Execute the code with automatic error fixing
        try:
            exec(code, exec_globals)
        except ValueError as e:
            if "could not convert string to float" in str(e):
                logger.warning(f"Type conversion error detected: {e}")
                # Try to fix common type conversion issues
                fixed_code = fix_type_conversion_errors(code)
                logger.info("Attempting to run code with automatic type fixes...")
                exec(fixed_code, exec_globals)
            else:
                raise
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            try:
                if obj is None:
                    return None
                elif hasattr(obj, 'item'):  # numpy scalar
                    value = obj.item()
                    # Handle NaN and infinity values
                    if isinstance(value, float):
                        if np.isnan(value):
                            return None
                        elif np.isinf(value):
                            return "infinity" if value > 0 else "-infinity"
                    return value
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, float):
                    # Handle regular Python float NaN and infinity
                    if np.isnan(obj):
                        return None
                    elif np.isinf(obj):
                        return "infinity" if obj > 0 else "-infinity"
                    return obj
                elif hasattr(obj, 'to_dict'):  # pandas objects like Series, DataFrame
                    return str(obj)  # Convert to string representation
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return convert_numpy_types(obj.tolist())
                else:
                    return obj
            except Exception as e:
                logger.warning(f"Could not convert object of type {type(obj)}: {e}")
                return str(obj)
        
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
        error_msg = str(e)
        logger.error(f"Code execution failed: {e}")
        logger.error(f"Code that failed:\n{code}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Provide more helpful error messages for common issues
        if "Can only use .str accessor with string values" in error_msg:
            error_msg = (
                "Error: Attempted to use .str accessor on non-string data. "
                "Make sure to check data types before using string methods. "
                "Use df[col].dtype == 'object' to check for string columns, "
                "or convert explicitly with df[col].astype(str) before using .str methods."
            )
        
        return {
            'success': False,
            'data_context': data_context,
            'result': None,
            'error': error_msg
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
- boto3 (for S3 access)
- json
- io
- re (for URL detection)

Available helper functions:
- detect_urls(text): Detects URLs in text and returns a list (supports HTTP, HTTPS, and S3 URLs)
- is_valid_url(url): Check if URL format is valid before attempting to access
- scrape_web_data(url, data_context): Scrapes tables from HTTP/HTTPS URLs and adds them to data_context
- download_from_s3(url, data_context): Downloads data from S3 URLs and loads into data_context
- download_data_from_url(url, data_context): Universal function that handles both HTTP and S3 URLs
- execute_duckdb_query(query, data_context): Execute SQL queries on DataFrames using DuckDB
- is_s3_url(url): Check if a URL is an S3 URL

Your capabilities:
1. **S3 Data Access**: Download CSV, JSON, Parquet, Excel files from S3 buckets using s3:// URLs
2. **Automatic URL Detection & Scraping**: Detect HTTP/HTTPS/S3 URLs in questions and automatically process data 
3. **Web Scraping**: Use the scrape_web_data() helper function or pandas.read_html() to scrape tables from websites
4. **Advanced SQL Queries**: Use DuckDB for complex SQL operations across multiple DataFrames
5. **File Processing**: Read CSV, JSON, Parquet, Excel and other data files using pandas
6. **Remote Parquet Files**: Query parquet files directly from S3 using DuckDB
7. **Data Loading**: Process uploaded files and create DataFrames

Code Templates:

# MANDATORY CODE TEMPLATE - Copy this exactly and do NOT add analysis code:
detected_urls = detect_urls(questions)
if detected_urls:
    for url in detected_urls:
        download_result = download_data_from_url(url, data_context)
        if download_result['success']:
            data_context = download_result['data_context']
            if 'scraped_tables' in download_result:
                # Web scraping result
                scraped_info = []
                for table_info in download_result['scraped_tables']:
                    scraped_info.append(f"{table_info['name']}: {table_info['shape']}")
                result = f"Successfully scraped {len(download_result['scraped_tables'])} tables: {', '.join(scraped_info)}"
            else:
                # S3 or other download result
                result = f"Successfully downloaded data from {url}"
        else:
            result = f"Failed to process {url}: {download_result['error']}"

# Direct web scraping example:
# tables = pd.read_html(url)
# # Clean data safely
# for i, table in enumerate(tables):
#     # Safe numeric conversion
#     for col in table.columns:
#         if table[col].dtype == 'object':
#             table[col] = pd.to_numeric(table[col], errors='coerce')
#     data_context[f'scraped_table_{i+1}'] = table

# S3 download example:
# s3_result = download_from_s3('s3://bucket-name/path/to/file.csv', data_context)
# if s3_result['success']:
#     data_context = s3_result['data_context']
#     print(f"Downloaded: {s3_result['filename']}")

# DuckDB query examples:
# Basic query on loaded DataFrames:
# query_result = execute_duckdb_query("SELECT * FROM data_csv LIMIT 10", data_context)
# if query_result['success']:
#     data_context = query_result['data_context']
#     print(f"Query result shape: {query_result['result_shape']}")

# Advanced DuckDB with joins:
# query = "SELECT a.col1, b.col2 FROM table1 a JOIN table2 b ON a.id = b.id"
# join_result = execute_duckdb_query(query, data_context)

# Direct S3 parquet query (requires DuckDB S3 extension):
# parquet_query = "SELECT * FROM 's3://bucket/path/file.parquet' LIMIT 100"
# parquet_result = execute_duckdb_query(parquet_query, data_context)

# File processing example:
# df = pd.read_csv('filename.csv')
# data_context['processed_data'] = df

CRITICAL RULES: 
- ONLY USE THE MANDATORY CODE TEMPLATE ABOVE - copy it exactly as written
- DO NOT ADD ANY DATA ANALYSIS OR PROCESSING CODE in this stage
- DO NOT access or modify individual columns or values
- DO NOT try to answer questions or perform calculations here  
- DO NOT use iloc, loc, filtering, or data manipulation
- ONLY scrape data and store it in data_context using the exact template
- Set 'result' variable to the scraping summary only
- Data analysis happens in the next stage, NOT here
- Your code should be complete and executable
- VIOLATION OF THESE RULES WILL CAUSE SYSTEM FAILURE

Write Python code to gather the required data. End your response with the code block only.
"""

DATA_ANALYSIS_PROMPT = """
You are a Data Analysis Specialist. Your task is to write Python code to perform statistical analysis and calculations on the available data.

CRITICAL DATA TYPE SAFETY RULES:
1. **ALWAYS check column data types before numerical operations**
2. **Use pd.to_numeric(column, errors='coerce') to safely convert strings to numbers**
3. **Check for NaN values after conversion and handle them appropriately**
4. **Never assume a column contains numeric data - always verify first**
5. **Use try-except blocks for operations that might fail due to data types**

MATPLOTLIB VISUALIZATION RULES:
6. **ALWAYS use PNG format for matplotlib figures: plt.savefig(buffer, format='png', dpi=80)**
7. **NEVER use quality parameter with plt.savefig() - it causes errors**
8. **Use "data:image/png;base64,..." for base64 encoded images**

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

# Correlation example (handling missing values and type conversion):
# df = data_context['filename.csv']  # Use actual filename from data_context keys
# # Safe type conversion - ALWAYS do this before numerical operations
# df = safe_numeric_convert(df, ['col1', 'col2'])
# # Or manually: df['col1'] = pd.to_numeric(df['col1'], errors='coerce')
# #              df['col2'] = pd.to_numeric(df['col2'], errors='coerce')
# # Drop rows where either column has missing values
# df_clean = df[['col1', 'col2']].dropna()
# if len(df_clean) > 1:
#     correlation = df_clean['col1'].corr(df_clean['col2'])
#     result['correlation'] = correlation
# else:
#     result['correlation'] = None

# Linear regression with safe data access:
# from sklearn.linear_model import LinearRegression
# df = data_context['scraped_table_1']  # Use actual table name
# # CRITICAL: Ensure both columns are numeric before any calculations
# df = safe_numeric_convert(df, ['x_column', 'y_column'])
# # Alternative: df['x_column'] = pd.to_numeric(df['x_column'], errors='coerce') 
# #              df['y_column'] = pd.to_numeric(df['y_column'], errors='coerce')
# # Drop rows where either column has missing values
# df_clean = df[['x_column', 'y_column']].dropna()
# if len(df_clean) > 1:
#     X = df_clean[['x_column']]
#     y = df_clean['y_column']
#     model = LinearRegression().fit(X, y)
#     result['slope'] = model.coef_[0]
#     result['intercept'] = model.intercept_
# else:
#     result['slope'] = None
#     result['intercept'] = None

# Safe data filtering example:
# df = data_context['scraped_table_1']
# # Convert to numeric safely
# df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')
# # Filter with safety check
# filtered = df[df['numeric_col'] > threshold]
# if len(filtered) > 0:
#     result_value = filtered.iloc[0]['some_column']
# else:
#     result_value = "No data found matching criteria"

# Statistical calculations:
# result['mean'] = df['column'].mean()
# result['median'] = df['column'].median()
# result['sum'] = df['column'].sum()

IMPORTANT:
- Access scraped data using data_context['scraped_table_1'] or appropriate scraped table name
- Store analysis results in a 'result' dictionary
- ALWAYS check if filtered data is empty before using iloc[0] - use len(filtered) > 0
- ALWAYS handle missing values by using .dropna() on the relevant columns before analysis
- ALWAYS ensure consistent sample sizes when doing correlations or regression
- Use pd.to_numeric(column, errors='coerce') for safe numeric conversion
- Use error handling for robust calculations and provide fallback values
- Check that you have enough data points (>1) before performing analysis
- When filtering data, always check if results exist before accessing with iloc
- NEVER use .str accessor without first checking column data type
- Check column dtypes with df[col].dtype before string operations
- Your code should be complete and executable

Write Python code to perform the required analysis. End your response with the code block only.
"""

PRESENTATION_PROMPT = """
You are a Data Presentation Specialist. Your task is to create a final response that directly answers the original questions with supporting visualizations.

GOAL: Structure the response as a question-answering system, not a generic data report.

Available imports and libraries:
- matplotlib.pyplot as plt (set to Agg backend)
- seaborn as sns
- pandas as pd
- base64 for encoding images
- json for output formatting
- io for handling image buffers

RESPONSE FORMAT:
Structure the result as a dictionary with:
- "questions": List of original questions
- "answers": List of direct answers corresponding to each question  
- "analysis_summary": Brief summary of analysis performed
- "visualizations": Charts that support the answers (if relevant)
- "data_insights": Key insights derived from the data
- "recommendations": Suggestions for more specific questions (if original questions were vague)

ANSWER EXAMPLES:

For question "What is the average sales?":
- Answer: "The average sales across all records is $X,XXX"
- Visualization: Bar chart showing sales distribution

For vague question "This is a test question":
- Answer: "This question is too general to provide specific insights. Based on the available data with columns [col1, col2], I can provide a basic data overview."
- Recommendation: "For more meaningful analysis, consider asking: 'What is the relationship between col1 and col2?' or 'What are the summary statistics for this dataset?'"

VISUALIZATION GUIDELINES:
- Only create charts that directly support your answers
- Use descriptive titles that relate to the questions
- Keep image sizes under 100kB
- Always use plt.close() after saving plots
- **CRITICAL**: Use PNG format only for matplotlib figures: plt.savefig(buffer, format='png', dpi=80)
- **NEVER** use quality parameter with matplotlib - it's not supported
- For base64 encoding: use "data:image/png;base64,..." format

DATA SAFETY RULES:
- NEVER use .str accessor on columns without first checking if they are string type
- Always check column dtypes before string operations: df[col].dtype == 'object'
- Convert to string explicitly if needed: df[col].astype(str) before using .str
- Handle mixed data types safely with pd.to_numeric() or astype()
- Use error handling for all data type operations

IMPORTANT:
- Directly address each original question in your answers
- If questions can't be answered with available data, explain why
- Provide alternative questions that could be answered
- Structure as question-answer pairs, not generic statistical output
- Set 'result' variable to the final structured dictionary
- NEVER use .str accessor without type checking first

Write Python code to create a question-focused response with supporting visualizations. End your response with the code block only.
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
    
    # Import datetime and setup debug directory at the start
    import os
    from datetime import datetime
    debug_dir = "logs/llm_debug"
    os.makedirs(debug_dir, exist_ok=True)

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
        
        # Check for URLs in questions and determine if we need data gathering
        detected_urls = detect_urls(questions)
        has_csv_files = any(filename.endswith('.csv') for filename in file_info)
        
        # Trigger data gathering if:
        # 1. No data context yet, OR
        # 2. URLs detected in questions, OR 
        # 3. Keywords indicating web scraping needed
        needs_additional_data = (
            not data_context or 
            detected_urls or
            any(keyword in questions.lower() for keyword in ['scrape', 'fetch', 'url', 'web', 'http'])
        )
        
        logger.info(f"URLs detected: {detected_urls}")
        logger.info(f"Has CSV files: {has_csv_files}")
        logger.info(f"Needs additional data: {needs_additional_data}")
        
        if needs_additional_data:
            # Prepare information for the gathering stage
            url_info = f"Detected URLs: {detected_urls}" if detected_urls else "No URLs detected"
            data_gathering_context = f"""
Questions:
{questions}

{url_info}
Files provided: {', '.join(file_info)}
Has CSV files: {has_csv_files}

Existing data context:
{data_summary}

INSTRUCTIONS:
- If URLs are detected and no CSV files provided, use detect_urls(questions) and scrape_web_data() to get data
- The 'questions' variable contains the full questions text for URL detection
- Use this pattern: 
  detected_urls = detect_urls(questions)
  for url in detected_urls:
      scrape_result = scrape_web_data(url, data_context)
      if scrape_result['success']:
          data_context = scrape_result['data_context']
- Store all scraped data in data_context with descriptive names
- Use safe data type handling to avoid conversion errors
- NOTE: Uploaded files are already available in data_context."""

            gathering_messages = [
                {"role": "system", "content": DATA_GATHERING_PROMPT},
                {"role": "user", "content": data_gathering_context}
            ]
            
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=gathering_messages,
                max_tokens=4096,  # Increase token limit for complete code generation
                temperature=0.2   # Lower temperature for more consistent code output
            )
            
            raw_response = response.choices[0].message.content
            logger.info(f"Raw LLM response for data gathering:\n{raw_response}")
            
            # Write raw response to debug file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"{debug_dir}/gathering_{timestamp}.txt"
            with open(debug_file, 'w') as f:
                f.write("=== RAW LLM RESPONSE FOR DATA GATHERING ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Question: {questions}\n")
                f.write("="*50 + "\n")
                f.write(raw_response)
                f.write("\n" + "="*50 + "\n")
            logger.info(f"Raw response saved to {debug_file}")
            
            gathering_code = extract_code_from_response(raw_response)
            logger.info(f"Extracted code (length: {len(gathering_code)}):\n{gathering_code}")
            
            if gathering_code:
                logger.info("Executing data gathering code...")
                # Create a copy of data_context and add questions for helper functions
                gathering_context = data_context.copy()
                gathering_context['questions'] = questions
                
                # Try to validate the code syntax before execution
                try:
                    compile(gathering_code, '<string>', 'exec')
                    logger.info("Code syntax validation passed")
                except SyntaxError as e:
                    logger.error(f"Syntax error in extracted code: {e}")
                    logger.error(f"Problematic code:\n{gathering_code}")
                    
                    # Try multiple fixes for common indentation issues
                    import textwrap
                    fixed_code = None
                    
                    # Fix 1: Simple dedent
                    try:
                        test_code = textwrap.dedent(gathering_code)
                        compile(test_code, '<string>', 'exec')
                        fixed_code = test_code
                        logger.info("Fixed with simple dedent")
                    except SyntaxError:
                        pass
                    
                    # Fix 2: Remove leading spaces from all lines
                    if not fixed_code:
                        try:
                            lines = gathering_code.split('\n')
                            # Find minimum indentation (excluding empty lines)
                            min_indent = float('inf')
                            for line in lines:
                                if line.strip():
                                    indent = len(line) - len(line.lstrip())
                                    min_indent = min(min_indent, indent)
                            
                            if min_indent > 0 and min_indent != float('inf'):
                                test_code = '\n'.join(line[min_indent:] if line.strip() else line 
                                                     for line in lines)
                                compile(test_code, '<string>', 'exec')
                                fixed_code = test_code
                                logger.info("Fixed by removing minimum indentation")
                        except (SyntaxError, ValueError):
                            pass
                    
                    if fixed_code:
                        gathering_code = fixed_code
                        logger.info("Indentation fixed successfully")
                    else:
                        logger.error(f"Could not fix syntax error: {e}")
                        return {"error": f"Data gathering stage failed: {e}"}
                
                exec_result = execute_python_code(gathering_code, gathering_context)
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
        
        # Get available dataset keys for the prompt
        available_keys_analysis = [f"'{key}'" for key in data_context.keys() if isinstance(data_context[key], pd.DataFrame)]
        keys_info_analysis = f"Available data_context keys: {', '.join(available_keys_analysis)}" if available_keys_analysis else "No DataFrame data available"
        
        analysis_messages = [
            {"role": "system", "content": DATA_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": f"Questions:\n{questions}\n\nAvailable data:\n" + 
                          "\n".join(data_info) +
                          f"\n\n{keys_info_analysis}\n\nWrite Python code to analyze the data and answer the questions. Use the EXACT column names shown above and access data using data_context[key] where key is one of the available keys."
            }
        ]
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=analysis_messages,
            max_tokens=4096,  # Increase token limit for complete code generation
            temperature=0.2   # Lower temperature for more consistent code output
        )
        
        raw_analysis_response = response.choices[0].message.content
        
        # Write raw analysis response to debug file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = f"{debug_dir}/analysis_{timestamp}.txt"
        with open(debug_file, 'w') as f:
            f.write("=== RAW LLM RESPONSE FOR ANALYSIS ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Question: {questions}\n")
            f.write("="*50 + "\n")
            f.write(raw_analysis_response)
            f.write("\n" + "="*50 + "\n")
        logger.info(f"Raw analysis response saved to {debug_file}")
        
        analysis_code = extract_code_from_response(raw_analysis_response)
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
        
        # Get available dataset keys for the prompt
        available_keys = [f"'{key}'" for key in data_context.keys() if isinstance(data_context[key], pd.DataFrame)]
        keys_info = f"Available data_context keys: {', '.join(available_keys)}" if available_keys else "No DataFrame data available"
        
        # Safely serialize analysis results
        def safe_json_serialize(obj):
            """Safely serialize objects to JSON-compatible format"""
            try:
                if obj is None:
                    return "None"
                elif isinstance(obj, (str, int, float, bool)):
                    return obj
                elif isinstance(obj, dict):
                    return {k: safe_json_serialize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [safe_json_serialize(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'to_dict'):  # pandas objects
                    return str(obj)
                else:
                    return str(obj)
            except Exception:
                return str(obj)
        
        try:
            analysis_results_str = json.dumps(safe_json_serialize(analysis_results), indent=2)
        except Exception as e:
            logger.warning(f"Failed to serialize analysis results: {e}")
            analysis_results_str = str(analysis_results)
        
        presentation_messages = [
            {"role": "system", "content": PRESENTATION_PROMPT},
            {
                "role": "user",
                "content": f"Questions:\n{questions}\n\nAnalysis results:\n{analysis_results_str}\n\nAvailable data:\n" +
                          "\n".join(data_info_detailed) +
                          f"\n\n{keys_info}\n\nIMPORTANT: \n- Use data_context[key] to access data where 'key' is one of the available keys listed above - NEVER read from files!\n- Store your final result in the 'result' variable as a Python dictionary\n- Do NOT call json.dumps() - just create the dictionary\n- Convert numpy types to Python types (use int() and float())\n\nWrite Python code to create visualizations and format the final result dictionary."
            }
        ]
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=presentation_messages,
            max_tokens=4096,  # Increase token limit for complete code generation
            temperature=0.2   # Lower temperature for more consistent code output
        )
        
        raw_presentation_response = response.choices[0].message.content
        
        # Write raw presentation response to debug file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = f"{debug_dir}/presentation_{timestamp}.txt"
        with open(debug_file, 'w') as f:
            f.write("=== RAW LLM RESPONSE FOR PRESENTATION ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Question: {questions}\n")
            f.write("="*50 + "\n")
            f.write(raw_presentation_response)
            f.write("\n" + "="*50 + "\n")
        logger.info(f"Raw presentation response saved to {debug_file}")
        
        presentation_code = extract_code_from_response(raw_presentation_response)
        if presentation_code:
            logger.info("Executing presentation code...")
            exec_result = execute_python_code(presentation_code, data_context)
            if exec_result['success']:
                final_result = exec_result['result']
                logger.info("Presentation completed successfully.")
                return final_result
            else:
                logger.error(f"Presentation failed: {exec_result['error']}")
                # Try to return analysis results as fallback
                if analysis_results:
                    logger.info("Falling back to analysis results after presentation failure")
                    return {
                        "questions": [questions],
                        "answers": ["Analysis completed but presentation failed. Raw results included."],
                        "analysis_summary": f"Error in presentation: {exec_result['error']}",
                        "raw_results": analysis_results
                    }
                return {"error": f"Presentation stage failed: {exec_result['error']}"}
        
        # If no presentation code was generated, format analysis results as a proper response
        if analysis_results:
            logger.info("No presentation code generated, formatting analysis results")
            return {
                "questions": [questions],
                "answers": ["Analysis completed. See results below."],
                "analysis_summary": "Data analysis performed successfully",
                "data_insights": analysis_results,
                "visualizations": []
            }
        
        # If we have data context but no analysis results, provide basic data summary
        if data_context:
            logger.info("No analysis results, providing data summary")
            data_summary_info = {}
            for name, df in data_context.items():
                if isinstance(df, pd.DataFrame):
                    data_summary_info[name] = {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
            
            return {
                "questions": [questions],
                "answers": ["Data loaded successfully. See summary below."],
                "analysis_summary": "Data has been loaded and is ready for analysis",
                "data_summary": data_summary_info,
                "recommendations": ["Please provide more specific questions for detailed analysis"]
            }
        
        # Final fallback - should rarely reach here
        logger.warning("No results generated at any stage")
        return {
            "questions": [questions],
            "answers": ["Unable to process the request. Please check your data and questions."],
            "analysis_summary": "Processing incomplete",
            "error_details": "No data or analysis results were generated"
        }
        
    except Exception as e:
        logger.error(f"Error in three-stage agent: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Three-stage agent failed: {str(e)}"}

def extract_code_from_response(content: str) -> str:
    """
    Extract Python code from LLM response.
    """
    import re
    import textwrap
    
    # Look for code blocks
    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        # Fix common indentation issues
        if code and not code.startswith(' ') and '\n' in code:
            # If first line has no indent but subsequent lines do, dedent the whole block
            lines = code.split('\n')
            if len(lines) > 1 and lines[1].startswith(' '):
                code = textwrap.dedent(code)
        return code
    
    # If no code blocks, assume the entire content might be code
    # First, check if it looks like Python code
    lines = content.split('\n')
    
    # Check if the content looks like code (has Python keywords, assignments, etc.)
    code_indicators = ['import ', 'def ', '=', 'if ', 'for ', 'while ', 'try:', 'except:', 
                      'data_context', 'result', 'detected_urls', 'scrape_result']
    
    # Count how many lines look like code
    code_like_lines = 0
    for line in lines:
        if any(indicator in line for indicator in code_indicators):
            code_like_lines += 1
    
    # If most lines look like code, treat the whole thing as code
    if code_like_lines >= len([l for l in lines if l.strip()]) * 0.5:
        code = content.strip()
        # Fix indentation issues
        if code and '\n' in code:
            first_line = code.split('\n')[0]
            # If first line has no indent but subsequent lines do, we need to handle it
            if not first_line.startswith(' '):
                lines = code.split('\n')
                # Check if this is a continuation that needs fixing
                if len(lines) > 1 and lines[1].strip() and not lines[1].startswith(' '):
                    # Lines are properly aligned, return as-is
                    return code
                elif len(lines) > 1 and lines[1].startswith(' '):
                    # We have indentation mismatch - need to align properly
                    # This is likely a fragment starting mid-block
                    return code
        return code
    
    # Otherwise, try to extract code portions
    code_started = False
    code_lines = []
    
    for line in lines:
        # Start collecting code after certain indicators
        if any(indicator in line.lower() for indicator in ['import ', 'def ', 'data_context', 'result =', 'url =', 'detected_urls']):
            code_started = True
        
        if code_started:
            # Stop if we hit a non-code line (e.g., explanatory text)
            if line.strip() and not line.startswith(' ') and not any(
                indicator in line.lower() for indicator in 
                ['=', 'import ', 'def ', 'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except:', '#', 'return', 'pass', 'break', 'continue']
            ):
                # Check if this looks like prose rather than code
                if len(line.split()) > 10 and '.' in line and not '"' in line and not "'" in line:
                    break
            code_lines.append(line)
    
    if code_lines:
        code = '\n'.join(code_lines).strip()
        # Fix indentation issues for extracted code
        if code and not code.startswith(' ') and '\n' in code:
            lines = code.split('\n')
            if len(lines) > 1 and lines[1].startswith(' '):
                code = textwrap.dedent(code)
        return code
    
    return ""
