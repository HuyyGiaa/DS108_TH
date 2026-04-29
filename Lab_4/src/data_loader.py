import pandas as pd
from pathlib import Path



def load_data(split: str):
    current_dir = Path(__file__).parent
    file_path = current_dir / '..' / 'data' / 'raw' / f'{split}.csv'
    if not file_path.exists():
        print(f"Không thể load được data: {file_path}")
        return None
    else:
        data = pd.read_csv(file_path)
    return data
    
    
