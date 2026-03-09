import pickle
import json
import pprint


def print_structure(obj, indent=0, max_depth=3):
    """Recursively print data structure"""
    if indent > max_depth:
        return

    prefix = "  " * indent
    if isinstance(obj, dict):
        print(f"{prefix}{{")
        for key, value in list(obj.items())[:5]:  # Show first 5 items
            if isinstance(value, (dict, list)):
                print(f"{prefix}  '{key}': {type(value).__name__}")
                print_structure(value, indent + 2, max_depth)
            else:
                print(f"{prefix}  '{key}': {str(value)[:50]}")
        if len(obj) > 5:
            print(f"{prefix}  ... ({len(obj) - 5} more items)")
        print(f"{prefix}}}")
    elif isinstance(obj, list):
        print(f"{prefix}[")
        for i, item in enumerate(obj[:3]):  # Show first 3 items
            print_structure(item, indent + 1, max_depth)
        if len(obj) > 3:
            print(f"{prefix}  ... ({len(obj) - 3} more items)")
        print(f"{prefix}]")
    else:
        print(f"{prefix}{type(obj).__name__}")


if __name__ == "__main__":
    # Replace 'path_to_your_file.pkl' with the actual path to your PKL file
    file_path = 'results.pkl'

    # Open the file in binary mode and load the data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Option 1: Pretty print with nested structure (basic)
    print_structure(data)

# time: 15.951970100402832, error: 0.2393538446217699