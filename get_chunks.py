import re
import json
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize



def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return sent_tokenize(text)


def create_chunks(text, chunk_size=1000):
    text = clean_text(text)
    sentences = split_sentences(text)

    chunks = []
    chunk_id = 0

    current_chunk = []
    current_length = 0

    sentence_start_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk size → save chunk
        if current_length + sentence_len > chunk_size and current_chunk:

            chunk_text = " ".join(current_chunk)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_char": text.find(current_chunk[0]),
                "end_char": text.find(current_chunk[-1]) + len(current_chunk[-1]),
                "sentence_start": sentence_start_idx,
                "sentence_end": i - 1
            })

            chunk_id += 1

            # 🔁 overlap: keep last sentence
            current_chunk = [current_chunk[-1]]
            current_length = len(current_chunk[0])
            sentence_start_idx = i - 1

        # add sentence
        current_chunk.append(sentence)
        current_length += sentence_len

    # last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_char": text.find(current_chunk[0]),
            "end_char": text.find(current_chunk[-1]) + len(current_chunk[-1]),
            "sentence_start": sentence_start_idx,
            "sentence_end": len(sentences) - 1
        })

    return chunks

# open data
with open("combined_document.json", "r") as document_data_file:
    document_data = json.load(document_data_file)

# create chunks
for text_data in document_data["text_blocks"]:
    text_data["chunks"] = create_chunks(text_data["content"] )


# create new document that includes chunks
new_data = json.dumps(document_data, indent=4)
with open("combined_document_with_chunks.json", "w") as result_file:
    result_file.write(new_data)

