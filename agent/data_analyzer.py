import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculates the Pearson correlation coefficient between two columns of a DataFrame.

    Args:
        df: The pandas DataFrame.
        col1: The name of the first column.
        col2: The name of the second column.

    Returns:
        The correlation coefficient as a float. Returns 0.0 on error.
    """
    try:
        logger.info(f"Calculating correlation between '{col1}' and '{col2}'")
        # Ensure columns are numeric, coercing errors will turn non-numerics into NaN
        df[col1] = pd.to_numeric(df[col1], errors='coerce')
        df[col2] = pd.to_numeric(df[col2], errors='coerce')

        # Drop rows where the numeric conversion failed
        df.dropna(subset=[col1, col2], inplace=True)

        if df.empty:
            logger.warning("DataFrame is empty after cleaning for correlation calculation.")
            return 0.0

        correlation = df[col1].corr(df[col2])
        logger.info(f"Calculated correlation: {correlation}")
        return correlation
    except Exception as e:
        logger.error(f"Could not calculate correlation for '{col1}' and '{col2}': {e}")
        return 0.0

def perform_linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Performs a simple linear regression.

    Args:
        df: The pandas DataFrame.
        x_col: The name of the independent variable column.
        y_col: The name of the dependent variable column.

    Returns:
        A dictionary with 'slope' and 'intercept'. Returns zeros on error.
    """
    try:
        logger.info(f"Performing linear regression for y='{y_col}' vs x='{x_col}'")
        # Prepare data, ensuring numeric types and no missing values
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df.dropna(subset=[x_col, y_col], inplace=True)

        if df.empty:
            logger.warning("DataFrame is empty after cleaning for regression.")
            return {"slope": 0.0, "intercept": 0.0}

        X = df[[x_col]]
        y = df[y_col]

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_

        logger.info(f"Linear regression result: slope={slope}, intercept={intercept}")
        return {"slope": slope, "intercept": intercept}
    except Exception as e:
        logger.error(f"Could not perform linear regression for '{y_col}' vs '{x_col}': {e}")
        return {"slope": 0.0, "intercept": 0.0}
