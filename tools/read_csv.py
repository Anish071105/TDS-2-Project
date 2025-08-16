import os
import pandas as pd

def read_csv_preview(file_path: str, nrows: int = 5) -> dict:
    """
    Reads CSV file and returns a preview (first nrows rows + header) as a CSV string.

    Returns a dictionary with:
    - "success": True/False
    - "content": CSV string if success else error message
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        df = pd.read_csv(file_path)
        print(f"CSV loaded: {file_path} with {len(df)} rows and {len(df.columns)} columns.")
        preview_df = df.head(nrows)
        csv_str = preview_df.to_csv(index=False)
        return {"success": True, "content": csv_str}
    except Exception as e:
        return {"success": False, "error": str(e)}
