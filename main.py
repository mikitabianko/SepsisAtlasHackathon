import pymupdf4llm
import json
import os
import re
from markdown_it import MarkdownIt

class MarkdownTreeConverter:
    def __init__(self, filename, exclude_headers=None):
        self.filename = filename
        self.parser = MarkdownIt()
        self.chunk_id = 0
        
        # Sections to drop (case-insensitive)
        # We now check these against both real headers and bolded paragraphs
        self.exclude_headers = [h.lower() for h in (exclude_headers or ["references", "bibliography", "acknowledgments"])]
        
        self.root = {
            "title": "Root",
            "level": 0,
            "content": [],
            "children": [],
            "metadata": {"filename": self.filename}
        }
        self.stack = [self.root]
        
        # State tracking
        self.skipping_section = False
        self.skip_level = 999

    def clean_text(self, text):
        # Remove page markers
        text = re.sub(r'\n\d+\s*\|\s*Page.*', '', text)
        return text.strip()

    def _is_pseudo_header(self, text):
        """
        Detects if a paragraph is likely a header, e.g., '**References**' or 'REFERENCES'
        """
        clean_val = text.strip().replace("**", "").lower()
        if clean_val in self.exclude_headers:
            return True, clean_val
        return False, None

    def _create_paragraph_object(self, text, page_num, is_table=False):
        self.chunk_id += 1
        return {
            "text": text,
            "metadata": {
                "filename": self.filename,
                "page": page_num,
                "chunk_id": self.chunk_id,
                "is_table": is_table
            }
        }

    def process_page(self, md_text, page_num):
        tokens = self.parser.parse(md_text)
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 1. Handle Standard Markdown Headers (# Header)
            if token.type == "heading_open":
                level = int(token.tag[1])
                title = tokens[i + 1].content
                title_clean = title.strip().lower()

                if title_clean in self.exclude_headers:
                    self.skipping_section = True
                    self.skip_level = level
                elif self.skipping_section and level <= self.skip_level:
                    self.skipping_section = False
                    self.skip_level = 999

                if not self.skipping_section:
                    new_node = {"title": title, "level": level, "content": [], "children": []}
                    while len(self.stack) > 1 and self.stack[-1]["level"] >= level:
                        self.stack.pop()
                    self.stack[-1]["children"].append(new_node)
                    self.stack.append(new_node)
                i += 2 

            # 2. Handle Paragraphs & Bold "Pseudo-headers"
            elif token.type == "paragraph_open":
                content = tokens[i + 1].content
                
                # Check if this paragraph is actually a bolded header (e.g. **References**)
                is_header_trigger, _ = self._is_pseudo_header(content)
                
                if is_header_trigger:
                    self.skipping_section = True
                    # Treat pseudo-headers as a high-level skip (level 2)
                    self.skip_level = 2 
                elif content.strip():
                    if not self.skipping_section:
                        is_table = "|" in content and "---" in content
                        para_obj = self._create_paragraph_object(content, page_num, is_table)
                        self.stack[-1]["content"].append(para_obj)
                i += 2

            elif token.type in ["fence", "table_open"]:
                if not self.skipping_section:
                    content = token.content.strip() if token.type == "fence" else "Table Data"
                    para_obj = self._create_paragraph_object(content, page_num, is_table=True)
                    self.stack[-1]["content"].append(para_obj)
                i += 1
            else:
                i += 1

def convert_pdf_to_nested_json(pdf_path, output_json_path):
    if not os.path.exists(pdf_path):
        return

    # Configuration
    EXCLUDE_LIST = ["references", "bibliography", "appendix"]

    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    converter = MarkdownTreeConverter(os.path.basename(pdf_path), exclude_headers=EXCLUDE_LIST)
    
    for page in pages:
        page_num = page["metadata"].get("page_number", 0)
        converter.process_page(converter.clean_text(page["text"]), page_num)

    if 'children' in converter.root:
        # Filter the children list to exclude the "References" section
        # The title in your JSON is formatted as "**References**"
        converter.root['children'] = [
            child for child in converter.root['children'] 
            if child.get('title') != "**References**" and child.get('title') != "References" and child.get('title') != "***References***" 
        ]
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(converter.root, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    convert_pdf_to_nested_json("./articles/Besen_2016.pdf", "./workspace/output_tree.json")