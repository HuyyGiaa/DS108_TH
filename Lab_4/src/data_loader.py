import pandas as pd
from pathlib import Path



def load_data_raw(split: str):
    current_dir = Path(__file__).parent
    file_path = current_dir / '..' / 'data' / 'raw' / f'{split}.csv'
    if not file_path.exists():
        print(f"Can not load data: {file_path}")
        return None
    else:
        data = pd.read_csv(file_path, low_memory=False)
    return data
    
    
def load_data_tidy(split: str):
    current_dir = Path(__file__).parent
    file_path = current_dir / '..' / 'data' / 'tidy' / f'{split}.csv'
    if not file_path.exists():
        print(f"Can not load data: {file_path}")
        return None
    else:
        data = pd.read_csv(file_path, low_memory=False)
    return data