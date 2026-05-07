import get_chunks
import json
import pandas as pd

# create json file with added chunks
# def set_chunks(filename : str, chunk_size : int):
#     with open(filename, 'r') as data_file:
#         data = json.load(data_file)

#     for kapitel in data["kapitel"]:
#         kapitel["chunks"] = get_chunks.get_chunks(kapitel["text"], chunk_size)

#     with open('processed_data.json', 'w') as processed_data_file:
#         json.dump(data, processed_data_file, indent=4)


# def get_text_from_output(filename : str):
#     with open(filename, 'r') as data_file:
#         data = json.load(data_file)

def extract_chunks(node, chunks):
    """
    Recursively traverses the JSON tree and extracts text items from 'content'.
    """
    # 1. Extract text items from the current node's content
    if "content" in node:
        for item in node["content"]:
            chunk_data = {
                "section": node.get("title", "Root"),
                "text": item.get("text", ""),
                "page": item.get("metadata", {}).get("page", ""),
                "chunk_id": item.get("metadata", {}).get("chunk_id", ""),
                "is_table": item.get("metadata", {}).get("is_table", False)
            }
            chunks.append(chunk_data)
    
    # 2. Recurse into child nodes
    if "children" in node:
        for child in node["children"]:
            extract_chunks(child, chunks)

def main(input_file, output_csv):
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Flatten the hierarchy into a list of dictionaries
    all_chunks = []
    extract_chunks(data, all_chunks)

    # Convert to a DataFrame for easy processing
    df = pd.DataFrame(all_chunks)

    # Save the flattened chunks to a CSV file
    df.to_csv(output_csv, index=False)
    
    # Optionally: Combine all text into a single string (one large chunk)
    full_text = "\n\n".join([c['text'] for c in all_chunks])
    with open('full_text_combined.txt', 'w') as f:
        f.write(full_text)

    print(f"Successfully processed {len(all_chunks)} chunks.")
    print(f"Flattened data saved to: {output_csv}")
    print("Combined text saved to: full_text_combined.txt")

if __name__ == "__main__":
    main('output_tree.json', 'combined_chunks.csv')
        
