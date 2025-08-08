import duckdb
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def run_duckdb_query(query: str, data_context: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Executes a DuckDB query. If a data_context is provided, it registers
    the pandas DataFrames within it as virtual tables that can be queried.

    Args:
        query: The SQL query string to be executed by DuckDB.
        data_context: A dictionary mapping table names to pandas DataFrames.

    Returns:
        A pandas DataFrame with the query results.
        Returns an empty DataFrame on error.
    """
    try:
        # Log a truncated version of the query for security and brevity
        logger.info(f"Running DuckDB query: {query[:150]}...")

        # Connect to an in-memory database. Each query runs in a fresh instance.
        con = duckdb.connect(database=':memory:', read_only=False)

        # If a data context is provided, register its DataFrames as tables
        if data_context:
            for name, df in data_context.items():
                if isinstance(df, pd.DataFrame):
                    con.register(name, df)
                    logger.info(f"Registered DataFrame '{name}' as a DuckDB virtual table.")

        # DuckDB can install and load extensions directly from the query string.
        result_df = con.execute(query).fetchdf()

        con.close()

        logger.info(f"DuckDB query executed successfully, returned DataFrame with shape {result_df.shape}")
        return result_df

    except Exception as e:
        # DuckDB can raise various errors, including parsing or execution errors.
        logger.error(f"DuckDB query failed: {e}")
        # Re-raise the exception so the orchestrator knows the tool call failed
        raise e
