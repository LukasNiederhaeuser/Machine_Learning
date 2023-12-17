import os
import pandas as pd

# Define folder
FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

def read_file(folder, filename, csv=True) -> pd.DataFrame:
    
    """
    Load data from the specified file in the given folder.
    
    Parameters:
        - folder (str): The folder where the file is located.
        - filename (str): The name of the file.
        - csv (bool): Whether the file is a CSV. Default is True.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    
    if csv:
        ending=".csv"
        data = pd.read_csv(os.path.join(FOLDER, folder, filename+ending), delimiter=",")
    else:
        raise ValueError("Currently this function is only able to read CSV files")

        
    return data