import get_chunks
import json

# create json file with added chunks
def set_chunks(filename : str, chunk_size : int):
    with open(filename, 'r') as data_file:
        data = json.load(data_file)

    for kapitel in data["kapitel"]:
        kapitel["chunks"] = get_chunks.get_chunks(kapitel["text"], chunk_size)

    with open('processed_data.json', 'w') as processed_data_file:
        json.dump(data, processed_data_file, indent=4)