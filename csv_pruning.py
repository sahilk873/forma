import pandas as pd
import re

def prune(file, columns):
    """
    Prune rows from a CSV file if the first element of the row contains a number in the specified columns array.
    
    :param file: Path to the CSV file.
    :param columns: List of numbers to check against.
    """
    # Read the CSV file
    df = pd.read_csv(file)
    
    columns =[i for i in range(0, 46) if i not in columns]

    # Compile a regex pattern to extract numbers from the first element
    pattern = re.compile(r"\d+")
    
    # Function to check if the first element contains a number in the columns array
    def should_delete(row):
        match = pattern.search(str(row[0]))
        return match and int(match.group()) in columns
    
    # Filter rows based on the condition
    df = df[~df.apply(should_delete, axis=1)]

    # Save the pruned DataFrame back to a new CSV file
    pruned_file = f"pruned_{file}"
    df.to_csv(pruned_file, index=False)
    
    return df
