def prune(file, columns):
    """
    Prune columns from a CSV file.
    
    :param file: Path to the CSV file.
    :param columns: List of columns to prune.
    """
    import pandas as pd

    columns = [i for i in range(0, 45) if i not in columns]

    # Read the CSV file
    df = pd.read_csv(file)
    
    # Drop the specified columns
    if "landmarks" in file:
        df = df[~df.iloc[:, 1].isin(columns)]
    else:
        df.drop(columns=[col for col in columns if col in df.columns], inplace=True)
    
    # Save the pruned DataFrame back to a new CSV file
    pruned_file = f"pruned_{file}"
    df.to_csv(pruned_file, index=False)
    
    return df