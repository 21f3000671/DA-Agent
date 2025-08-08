import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def scrape_table_from_url(url: str) -> pd.DataFrame:
    """
    Scrapes the first HTML table found at a given URL using pandas.

    Args:
        url: The URL of the webpage to scrape.

    Returns:
        A pandas DataFrame containing the data from the first table found.
        Returns an empty DataFrame if no tables are found or an error occurs.
    """
    try:
        logger.info(f"Scraping table from URL: {url}")
        # Using pandas' read_html is efficient for finding tables.
        # It requires 'lxml' to be installed for the 'lxml' flavor.
        tables = pd.read_html(url, flavor='lxml')

        if not tables:
            logger.warning(f"No tables found at {url}")
            return pd.DataFrame()

        # The prompt implies we're interested in the main table on the page,
        # which is typically the first one.
        df = tables[0]
        logger.info(f"Successfully scraped a table with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to scrape table from {url}: {e}")
        # Return an empty DataFrame to prevent downstream errors.
        return pd.DataFrame()
