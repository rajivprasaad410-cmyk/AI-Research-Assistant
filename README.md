# AI Research Assistant using RAG

An AI-powered academic research assistant that allows users to upload research papers, index their content, and ask questions using Retrieval-Augmented Generation (RAG). The system retrieves relevant sections from the uploaded papers and generates answers with source and page references.

## Overview

Reading and understanding research papers can be time-consuming, especially when searching for specific concepts, methods, or results.

This project provides a simple interface where users can:

* Upload academic research papers in PDF format
* Extract text while preserving the document layout
* Convert document content into vector embeddings
* Store the embeddings in ChromaDB
* Ask questions about the indexed papers
* Retrieve the most relevant passages
* Generate answers using an LLM
* View the retrieved source passages and similarity scores

## System Workflow

```text
Academic PDF
     ↓
PDF Parsing
     ↓
Layout-Aware Text Extraction
     ↓
Document Chunking
     ↓
BGE Embeddings
     ↓
ChromaDB Vector Store
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Groq LLM
     ↓
AI-Generated Answer
     ↓
Source Citations
```

## Key Features

### 1. Academic PDF Upload

Users can upload research papers directly through the Streamlit interface.

### 2. Layout-Aware PDF Processing

The system uses PyMuPDF4LLM to extract content from PDFs while retaining useful document structure such as multi-column layouts, tables, and mathematical content.

Each page is converted into a LlamaIndex `Document` with metadata containing the original file name and page number.

### 3. Document Chunking

The extracted documents are divided into smaller chunks using LlamaIndex's `SentenceSplitter`.

* Chunk size: 512
* Chunk overlap: 50

This allows the retrieval system to find smaller and more relevant portions of a research paper.

### 4. Local Vector Database

The project uses ChromaDB as a persistent local vector database.

The document chunks are converted into embeddings and stored in a collection named:

```text
research_papers
```

Previously indexed versions of the same document are removed before re-indexing.

### 5. BGE Embeddings

The project uses the following Hugging Face embedding model:

```text
BAAI/bge-small-en-v1.5
```

The embeddings are generated locally, avoiding the need for a separate paid embedding API.

### 6. Retrieval-Augmented Generation

When a user asks a question:

1. The question is converted into an embedding.
2. ChromaDB searches for similar document chunks.
3. The top relevant chunks are retrieved.
4. The retrieved content is passed as context to the LLM.
5. The LLM generates an answer based only on the retrieved context.

The default retrieval value is Top-3 chunks, and the Streamlit interface allows the user to change the Top-K value from 1 to 6.

### 7. Groq LLM

The retrieved context is passed to a Groq-hosted LLM.

The current implementation uses:

```text
qwen/qwen3.6-27b
```

The prompt instructs the model to answer using only the retrieved context and provide document and page references where applicable.

### 8. Source References

For every generated response, the application can display:

* Source document
* Page number
* Similarity score
* Retrieved passage

This helps users understand where the generated answer came from.

## Technologies Used

| Technology  | Purpose                               |
| ----------- | ------------------------------------- |
| Python      | Core programming language             |
| Streamlit   | Web interface                         |
| LlamaIndex  | Document indexing and retrieval       |
| ChromaDB    | Vector database                       |
| BAAI BGE    | Text embeddings                       |
| Groq        | Large Language Model inference        |
| PyMuPDF4LLM | Layout-aware PDF extraction           |
| NLTK        | Natural language processing utilities |
| PyTorch     | Machine learning backend              |

## Project Structure

```text
AI-Research-Assistant/
│
├── app.py
│
├── requirements.txt
│
├── src/
│   ├── parser.py
│   ├── indexer.py
│   └── rag.py
│
└── .gitignore
```

### `app.py`

The main Streamlit application.

It handles:

* PDF uploading
* Document indexing
* Chat interface
* Retrieval parameter selection
* Conversation history
* Displaying retrieved sources

### `src/parser.py`

Responsible for extracting content from academic PDFs.

It:

* Reads the PDF
* Extracts page-level content
* Preserves useful layout information
* Creates LlamaIndex documents
* Adds file name and page number metadata

### `src/indexer.py`

Responsible for creating the vector index.

It:

* Parses the PDF
* Splits documents into chunks
* Generates BGE embeddings
* Stores vectors in ChromaDB
* Removes previous entries of the same document

### `src/rag.py`

Responsible for the question-answering pipeline.

It:

* Loads the Groq API key
* Initializes the Groq LLM
* Connects to ChromaDB
* Retrieves relevant document chunks
* Builds the RAG query engine
* Generates answers from retrieved context

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rajivprasaad410-cmyk/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## API Key Configuration

The application requires a Groq API key.

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

The application can also use Streamlit Secrets when deployed on Streamlit Cloud.

## Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

### Step 1: Upload a Research Paper

Use the sidebar to upload an academic PDF.

### Step 2: Index the Document

Click:

```text
Index Document into ChromaDB
```

The system will:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding Generation
 ↓
ChromaDB Storage
```

### Step 3: Ask Questions

Enter a question related to the uploaded research paper.

For example:

```text
What is the main mechanism proposed in the paper?
```

or:

```text
Explain the attention mechanism used in this paper.
```

### Step 4: View Retrieved Sources

The application displays the relevant passages used to generate the answer along with their similarity scores and page information.

## Why RAG?

A normal LLM may not have access to the contents of a user's research paper.

RAG solves this problem by providing the LLM with relevant information retrieved directly from the uploaded documents.

Instead of:

```text
Question → LLM → Answer
```

this project uses:

```text
Question
   ↓
Vector Search
   ↓
Relevant Paper Sections
   ↓
LLM + Retrieved Context
   ↓
Answer
```

This makes the system more suitable for question answering over specific research papers.

## Example

Suppose a user uploads a research paper about Transformer architecture and asks:

```text
Why is self-attention used in the proposed architecture?
```

The system searches the indexed paper for the most relevant chunks, retrieves the relevant sections, and provides those sections to the LLM as context.

The LLM then generates an answer based on the retrieved content instead of answering solely from its general knowledge.

## Advantages

* Simple interface for academic paper analysis
* Supports PDF research papers
* Uses semantic search instead of simple keyword matching
* Local ChromaDB storage
* Local embedding generation
* Source and page-level metadata
* Adjustable retrieval Top-K
* Reduces unsupported answers by restricting the LLM to retrieved context

## Future Improvements

* Support multiple research papers simultaneously
* Add paper comparison functionality
* Add automatic research-paper summarization
* Add citation export
* Add conversation memory across sessions
* Add research-paper metadata extraction
* Add hybrid keyword and semantic search
* Add graphical visualization of related papers and concepts

## Author

**Rajiv Prasaad**

GitHub:
https://github.com/rajivprasaad410-cmyk

## License

This project is intended for educational and research purposes.
