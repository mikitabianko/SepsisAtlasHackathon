import re
import json
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize



def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return sent_tokenize(text)


def get_chunks(
    text,
    target_chunk_size=1000,
    overlap_size=200,
    min_chunk_size=300
):

    text = clean_text(text)
    sentences = split_sentences(text)

    chunks = []
    chunk_id = 0

    current_sentences = []
    current_length = 0

    sentence_start_idx = 0

    for i, sentence in enumerate(sentences):

        sentence_len = len(sentence)
        extra_space = 1 if current_sentences else 0

        # save chunk if limit exceeded
        if (
            current_length + sentence_len + extra_space
            > target_chunk_size
            and current_sentences
        ):

            chunk_text = " ".join(current_sentences)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_char": text.find(chunk_text),
                "end_char": text.find(chunk_text) + len(chunk_text),
                "sentence_start": sentence_start_idx,
                "sentence_end": i - 1
            })

            chunk_id += 1

            # create overlap
            overlap_sentences = []
            overlap_length = 0

            for s in reversed(current_sentences):

                if overlap_length + len(s) > overlap_size:
                    break

                overlap_sentences.insert(0, s)
                overlap_length += len(s)

            current_sentences = overlap_sentences
            current_length = overlap_length

            sentence_start_idx = i - len(overlap_sentences)

        # add sentence
        current_sentences.append(sentence)
        current_length += sentence_len + extra_space

    # final chunk
    if current_sentences:

        chunk_text = " ".join(current_sentences)

        final_chunk = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_char": text.find(chunk_text),
            "end_char": text.find(chunk_text) + len(chunk_text),
            "sentence_start": sentence_start_idx,
            "sentence_end": len(sentences) - 1
        }

        # =========================
        # Prevent tiny last chunk
        # =========================

        if (
            len(chunk_text) < min_chunk_size
            and len(chunks) > 0
        ):

            # merge into previous chunk
            chunks[-1]["text"] += " " + chunk_text
            chunks[-1]["end_char"] = final_chunk["end_char"]
            chunks[-1]["sentence_end"] = final_chunk["sentence_end"]

        else:
            chunks.append(final_chunk)

    return chunks

