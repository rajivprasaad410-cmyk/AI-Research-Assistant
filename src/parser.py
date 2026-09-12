import os
from typing import List, Optional
import pymupdf4llm
from llama_index.core.schema import Document

def parse_pdf_to_documents(pdf_path: str, original_filename: Optional[str] = None) -> List[Document]:
    """
    Parses a multi-column PDF into a list of LlamaIndex Document objects
    preserving tables, LaTeX formatting, and page-level metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # Extract Markdown chunks per page with layout retention
    page_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    
    documents = []
    # Retain the genuine user-facing filename instead of temp filenames
    file_name = original_filename if original_filename else os.path.basename(pdf_path)

    for item in page_data:
        text = item.get("text", "").strip()
        if not text:
            continue
            
        page_num = item.get("metadata", {}).get("page_number", 1)
        
        # Wrap each page into a LlamaIndex Document with metadata for citations
        doc = Document(
            text=text,
            metadata={
                "file_name": file_name,
                "page_number": page_num,
                "source": f"{file_name} (Page {page_num})"
            }
        )
        documents.append(doc)

    return documents

if __name__ == "__main__":
    test_pdf = os.path.join("data", "attention.pdf")
    docs = parse_pdf_to_documents(test_pdf)
    
    print(f"Successfully extracted {len(docs)} pages.")
    print("\n--- Preview of Page 1 Metadata ---")
    print(docs[0].metadata)
    print("\n--- First 300 Characters of Page 1 Content ---")
    print(docs[0].text[:300])