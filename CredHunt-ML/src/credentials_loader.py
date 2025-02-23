import csv
import os
import logging

def load_credentials(csv_filename="synthetic_credentials_snippets.csv"):
    """
    Load credentials from the CSV file.
    Each row is expected to have: Snippet Code, Value, Type.
    Returns a list of dictionaries.
    """
    credentials = []
    csv_path = os.path.join(os.path.dirname(__file__), "..", csv_filename)
    try:
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                credentials.append({
                    "snippet_code": row["Snippet Code"],
                    "value": row["Value"],
                    "type": row["Type"]
                })
        logging.info(f"Successfully loaded {len(credentials)} credentials from {csv_filename}")
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_path}")
        return []  # Return an empty list
    except Exception as e:
        logging.error(f"Error loading credentials from {csv_filename}: {str(e)}")
        return []  # Return an empty list
    return credentials

if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    creds = load_credentials()
    if creds:
        for c in creds[:5]:
            print(c)
    else:
        print("No credentials loaded.")