# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DA-Agent is a data analysis agent that combines OpenAI's GPT-4o with specialized tools for web scraping, database querying, statistical analysis, and visualization. The system accepts questions via text files and optional CSV data files, then uses AI-driven tool selection to analyze data and generate insights.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Run the FastAPI server locally
uv run uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test the API endpoint locally
uv run python test_endpoint.py

# Run a live test with sample data
uv run python run_live_test.py
```

### Deployment
The application is configured for Vercel deployment via `vercel.json` with the API endpoint at `api/index.py`.

## Architecture

### Core Components

**API Layer** (`api/index.py`):
- FastAPI application with a single `/api/` endpoint
- Accepts multipart file uploads (questions.txt + optional CSV files)
- Orchestrates the agent workflow and returns JSON responses

**Agent Orchestrator** (`agent/main.py`):
- Main agent logic using OpenAI's function calling API
- Manages conversation history and tool execution loops
- Maintains a `data_context` dictionary for DataFrames
- Handles tool result processing and error management

**Tool Modules**:
- `web_scraper.py`: HTML table scraping using pandas and BeautifulSoup
- `db_querier.py`: SQL queries via DuckDB with DataFrame integration
- `data_analyzer.py`: Statistical analysis (correlation, linear regression)
- `plotter.py`: Matplotlib/Seaborn visualization with base64 encoding

**Advanced Analysis Capabilities**:
- Multi-stage analysis pipeline: data gathering → analysis → presentation
- Dynamic code generation for custom data transformations
- Automatic syntax validation and indentation fixing for generated code
- Support for complex analytical workflows via Python code execution

### Data Flow

1. **Input Processing**: Files uploaded to `/api/` endpoint
2. **Context Loading**: CSV files loaded into pandas DataFrames in `data_context`
3. **AI Planning**: GPT-4o analyzes questions and available data
4. **Tool Execution**: Agent calls appropriate tools based on AI decisions
5. **Result Aggregation**: Tool outputs processed and stored in `data_context`
6. **Final Response**: AI generates structured JSON response with insights

### Tool Integration Pattern

Each tool follows a consistent pattern:
- Functions accept pandas DataFrames and return structured data
- The orchestrator manages DataFrame references via `data_context` dictionary
- Tool results are automatically stored with unique identifiers
- Error handling prevents cascade failures

### Key Dependencies

- **OpenAI**: GPT-4o for intelligent tool selection and analysis
- **FastAPI**: Web API framework
- **DuckDB**: In-memory SQL processing for large datasets
- **Pandas**: Core data manipulation
- **Scikit-learn**: Statistical analysis
- **Matplotlib/Seaborn**: Data visualization
- **BeautifulSoup**: Web scraping

## Configuration

### Environment Variables

**Core Configuration:**
- `OPENAI_API_KEY` or `LLM_API_KEY`: Required for LLM API access (loaded via python-dotenv)

**LLM Provider Configuration:**
- `LLM_PROVIDER`: LLM provider to use (default: "openai")
- `LLM_MODEL`: Model name to use (default: "gpt-4o")  
- `LLM_BASE_URL`: Custom endpoint URL (optional, for self-hosted or alternative providers)
- `LLM_API_KEY`: API key for the LLM provider (falls back to OPENAI_API_KEY if not set)

### File Structure
```
/agent/          # Core agent modules
/api/            # FastAPI web interface
main.py          # Simple CLI entry point
pyproject.toml   # Python dependencies and project config
vercel.json      # Vercel deployment configuration
```

## Development Notes

- The agent is limited to 10 conversation turns to prevent infinite loops
- Tool functions should handle errors gracefully and return safe defaults
- DataFrame operations automatically handle data type coercion and missing values
- Base64 image encoding is used for plot visualization in JSON responses
- DuckDB queries can reference any DataFrame in the `data_context` as virtual tables

### Error Handling and Fallbacks

- **Code Validation**: Generated Python code is syntax-checked before execution
- **Automatic Indentation Fixing**: Common indentation issues are automatically corrected
- **Graceful Degradation**: Multiple fallback levels ensure meaningful responses:
  1. If presentation fails → returns raw analysis results
  2. If analysis fails → returns data summary with column info
  3. If data loading fails → provides error diagnostics
- **Result Preservation**: Intermediate results are preserved across pipeline stages