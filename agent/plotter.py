import matplotlib
# Use a non-interactive backend to prevent issues in a server environment
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import logging

logger = logging.getLogger(__name__)

def create_scatterplot_with_regression(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str
) -> str:
    """
    Creates a scatterplot with a regression line and returns it as a base64 data URI.

    Args:
        df: The pandas DataFrame containing the data.
        x_col: The column name for the x-axis.
        y_col: The column name for the y-axis.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.

    Returns:
        A base64 encoded data URI string of the plot. Returns an empty string on error.
    """
    try:
        logger.info(f"Creating scatterplot for y='{y_col}' vs x='{x_col}'")

        # Ensure data is numeric
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df.dropna(subset=[x_col, y_col], inplace=True)

        if df.empty:
            logger.warning("DataFrame is empty after cleaning for plotting.")
            return ""

        plt.figure(figsize=(8, 6))

        # Use seaborn's regplot to easily create a scatter plot with a regression line
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'linestyle': 'dotted'} # As per prompt
        )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

        # Save the plot to an in-memory binary buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()  # Close the plot figure to free up memory

        img_buffer.seek(0)

        # Encode the binary data to a base64 string
        base64_encoded_img = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Format as a data URI
        data_uri = f"data:image/png;base64,{base64_encoded_img}"

        # Check if the generated URI is under the size limit
        if len(data_uri) > 100000:
            logger.warning(
                f"Generated plot data URI size ({len(data_uri)} bytes) "
                "exceeds the 100kB limit. The image might be rejected."
            )
            # In a real-world scenario, we might try to re-compress the image here.

        logger.info("Successfully created and encoded plot as a data URI.")
        return data_uri

    except Exception as e:
        logger.error(f"Failed to create plot for '{y_col}' vs '{x_col}': {e}")
        return ""
