import json
import os

def get_structure(obj, indent=0, out=None, max_list_preview=1):
    """
    Recursively inspects the structure of a JSON object and logs the type hierarchy.
    """
    if out is None:
        out = []

    prefix = "  " * indent

    if isinstance(obj, dict):
        out.append(f"{prefix}dict with {len(obj)} keys:")
        for key, value in obj.items():
            out.append(f"{prefix}  '{key}': {type(value).__name__}")
            get_structure(value, indent + 2, out, max_list_preview)

    elif isinstance(obj, list):
        out.append(f"{prefix}list with {len(obj)} elements")
        if obj and max_list_preview > 0:  # Only inspect first few
            get_structure(obj[0], indent + 1, out, max_list_preview - 1)

    else:
        out.append(f"{prefix}{type(obj).__name__}")

    return out


def inspect_json_file(file_path):
    """
    Detects whether the file is JSON or JSONL, loads sample data, 
    and returns a readable type structure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    out = []

    # JSONL → read first line only for preview
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            try:
                data = json.loads(first_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in first line: {e}")
        out.append("Detected JSONL file (showing structure of first record):")
        out.extend(get_structure(data))

    # JSON → read whole file
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")
        out.append("Detected JSON file:")
        out.extend(get_structure(data))

    else:
        raise ValueError("File must have a .json or .jsonl extension")

    return "\n".join(out)


# Example usage
if __name__ == "__main__":
    path = "dummy.json"  # change to your file
    structure_description = inspect_json_file(path)
    print(structure_description)