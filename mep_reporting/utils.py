# Generic formula for converting columns to snake_case
def to_snake_case_inplace(df): 
    """Convert column names of a DataFrame to snake_case inplace."""
    df.columns = (
        df.columns
        .str.replace(r'\(.*?\)', '', regex=True)  # Remove content inside parentheses
        .str.strip()  # Remove extra spaces left after removing parentheses
        .str.lower()  # Convert to lowercase
        .str.replace(r'[^a-z0-9]+', '_', regex=True)  # Replace non-alphanumeric characters with underscores
        .str.strip('_')  # Remove leading/trailing underscores
    )