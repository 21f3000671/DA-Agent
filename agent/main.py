import openai
import json
import pandas as pd
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assuming UploadFile is a type, for type hinting. In practice, we get the real object.
from fastapi import UploadFile

# Import the tool functions from the other modules
from . import web_scraper, data_analyzer, plotter, db_querier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client.
# It will automatically look for the OPENAI_API_KEY environment variable.
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

# Define the tools in the format the OpenAI API expects
tools = [
    {
        "type": "function",
        "function": {
            "name": "scrape_table_from_url",
            "description": "Scrapes the first HTML table from a given URL. Use this to get data from a webpage.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL of the webpage to scrape."}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_duckdb_query",
            "description": "Executes a DuckDB SQL query, useful for large datasets. Returns a pandas DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The DuckDB SQL query to execute."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_correlation",
            "description": "Calculates the Pearson correlation between two columns of a DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "df_name": {"type": "string", "description": "The name of the DataFrame to use (e.g., 'scraped_data', or a filename like 'data.csv')."},
                    "col1": {"type": "string", "description": "The first column name."},
                    "col2": {"type": "string", "description": "The second column name."},
                },
                "required": ["df_name", "col1", "col2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_linear_regression",
            "description": "Performs a simple linear regression on two columns of a DataFrame and returns the slope and intercept.",
             "parameters": {
                "type": "object",
                "properties": {
                    "df_name": {"type": "string", "description": "The name of the DataFrame to use."},
                    "x_col": {"type": "string", "description": "The independent variable column."},
                    "y_col": {"type": "string", "description": "The dependent variable column."},
                },
                "required": ["df_name", "x_col", "y_col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_scatterplot_with_regression",
            "description": "Creates a scatterplot with a dotted red regression line and returns a base64 data URI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "df_name": {"type": "string", "description": "The name of the DataFrame to use."},
                    "x_col": {"type": "string", "description": "The column for the x-axis."},
                    "y_col": {"type": "string", "description": "The column for the y-axis."},
                    "title": {"type": "string", "description": "The title for the plot."},
                    "xlabel": {"type": "string", "description": "The label for the x-axis."},
                    "ylabel": {"type": "string", "description": "The label for the y-axis."},
                },
                "required": ["df_name", "x_col", "y_col", "title", "xlabel", "ylabel"],
            },
        },
    },
]

# Map tool names to their actual functions
available_tools = {
    "scrape_table_from_url": web_scraper.scrape_table_from_url,
    "run_duckdb_query": db_querier.run_duckdb_query,
    "calculate_correlation": data_analyzer.calculate_correlation,
    "perform_linear_regression": data_analyzer.perform_linear_regression,
    "create_scatterplot_with_regression": plotter.create_scatterplot_with_regression,
}

# The main agent orchestrator function
async def run_agent(questions: str, files: List[UploadFile]) -> Any:
    if not client:
        return {"error": "OpenAI client is not initialized. Please check API key."}

    logger.info("Starting data agent...")

    # This dictionary will hold our data, like dataframes from scraped pages or uploaded files.
    data_context: Dict[str, pd.DataFrame] = {}

    # Read uploaded files into pandas DataFrames if they are CSVs.
    file_info = []
    for file in files:
        filename = file.filename
        file_info.append(filename)
        if filename.endswith('.csv'):
            try:
                content = await file.read()
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                data_context[filename] = df
                logger.info(f"Loaded uploaded CSV '{filename}' into a DataFrame.")
            except Exception as e:
                logger.error(f"Failed to read uploaded CSV file {filename}: {e}")

    # The conversation history with the LLM
    messages = [
        {
            "role": "system",
            "content": "You are a powerful data analyst agent. Your task is to answer the user's questions. "
                       "You have access to a set of tools for scraping, database querying, analysis, and plotting. "
                       "Follow these steps:\n"
                       "1. Understand the user's question and the provided files.\n"
                       "2. Formulate a plan and decide which tool to use.\n"
                       "3. Call the necessary tools with the correct parameters. When using a tool that needs a dataframe, "
                       "refer to it by its name in the `data_context` (e.g., 'scraped_data' or 'my_data.csv').\n"
                       "4. The results of tool calls will be dataframes or other values. You can then use other tools on this data.\n"
                       "5. IMPORTANT: If a tool call results in an error, stop immediately and report the error. Do not proceed with the plan.\n"
                       "6. When you have all the information, provide the final answer in the precise JSON format requested by the user. "
                       "Do not say anything else, just the final JSON."
        },
        {
            "role": "user",
            "content": f"Here are my questions:\n\n{questions}\n\nAnd here are the files I've provided: {', '.join(file_info)}"
        }
    ]

    # The main loop for the agent's conversation with the LLM
    for _ in range(10):  # Limit to 10 iterations to prevent infinite loops
        logger.info("Sending request to LLM...")

        response = client.chat.completions.create(
            model="gpt-4o",  # Using a powerful model capable of tool use
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # If there are no tool calls, the LLM has decided to give the final answer.
            logger.info("LLM provided final answer.")
            try:
                # The response content should be the final JSON
                final_answer = json.loads(response_message.content)
                return final_answer
            except json.JSONDecodeError:
                logger.error("Failed to decode the final JSON answer from the LLM.")
                return {"error": "LLM did not return valid JSON.", "content": response_message.content}

        # The LLM wants to call one or more tools
        messages.append(response_message)
        logger.info(f"LLM requested to call tools: {[tc.function.name for tc in tool_calls]}")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)

            if not function_to_call:
                logger.error(f"LLM tried to call an unknown function: {function_name}")
                continue

            try:
                function_args = json.loads(tool_call.function.arguments)

                # Special handling for functions that require a DataFrame from our context
                if "df_name" in function_args:
                    df_name = function_args.pop("df_name")
                    if df_name not in data_context:
                        raise ValueError(f"DataFrame '{df_name}' not found in data context.")
                    function_args['df'] = data_context[df_name]

                # Special handling for DuckDB to provide the full data context
                if function_name == 'run_duckdb_query':
                    function_args['data_context'] = data_context

                # Call the actual tool function
                function_response = function_to_call(**function_args)

                logger.info(f"Successfully executed tool '{function_name}'")

                # If the function returns a DataFrame, save it to our context
                # and give the LLM a summary instead of the whole DataFrame.
                tool_output_for_llm = ""
                if isinstance(function_response, pd.DataFrame):
                    # Give the result a unique name in the context
                    new_df_name = f"{function_name}_result_{tool_call.id[:4]}"
                    data_context[new_df_name] = function_response
                    tool_output_for_llm = (f"Tool '{function_name}' executed successfully and returned a DataFrame with "
                                           f"shape {function_response.shape}. It is now available as '{new_df_name}'.\n"
                                           f"Here are the first 3 rows:\n{function_response.head(3).to_string()}")
                else:
                    # For other return types (string, dict, float), serialize to string for the LLM
                    tool_output_for_llm = json.dumps({ "result": function_response }, default=str)

                # Append the tool's output to the conversation history
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output_for_llm,
                    }
                )

            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {e}")
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": str(e)}),
                    }
                )

    return {"error": "Agent exceeded maximum iterations."}
