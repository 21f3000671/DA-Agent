import duckdb
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def run_duckdb_query(query: str) -> pd.DataFrame:
    """
    Executes a DuckDB query, including potential extension loading,
    and returns the result as a pandas DataFrame.

    Args:
        query: The SQL query string to be executed by DuckDB.

    Returns:
        A pandas DataFrame with the query results.
        Returns an empty DataFrame on error.
    """
    try:
        # Log a truncated version of the query for security and brevity
        logger.info(f"Running DuckDB query: {query[:150]}...")

        # Connect to an in-memory database. Each query runs in a fresh instance.
        con = duckdb.connect(database=':memory:', read_only=False)

        # DuckDB can install and load extensions directly from the query string.
        # For example: 'INSTALL httpfs; LOAD httpfs;'
        result_df = con.execute(query).fetchdf()

        con.close()

        logger.info(f"DuckDB query executed successfully, returned DataFrame with shape {result_df.shape}")
        return result_df

    except Exception as e:
        # DuckDB can raise various errors, including parsing or execution errors.
        logger.error(f"DuckDB query failed: {e}")
        return pd.DataFrame()
