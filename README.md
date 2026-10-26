# DocuMind — PDF-Based RAG Application

DocuMind is a Python-based application that explores Retrieval-Augmented Generation (RAG) to answer questions using information extracted from PDF documents.

The project demonstrates a document-processing pipeline involving PDF text extraction, text chunking, embedding generation, semantic retrieval, and answer generation.

## Features

- **PDF Text Extraction:** Extract text from PDF documents.
- **Text Chunking:** Divide extracted text into smaller, manageable chunks.
- **Embedding Generation:** Convert text chunks into numerical vector representations.
- **Semantic Retrieval:** Retrieve relevant document content for a query.
- **Answer Generation:** Generate responses using retrieved context.
- **Retrieval Evaluation:** Evaluate the performance of the retrieval pipeline.
- **Testing:** Includes tests for the answer-generation pipeline.

## Tech Stack

- Python
- Retrieval-Augmented Generation (RAG)
- Text Embeddings
- Semantic Search
- PDF Processing
- Pytest

*Specific libraries and model providers will be documented as the implementation is verified.*

## Project Structure

```text
DocuMind/
├── app.py
├── answer.py
├── chunk_text.py
├── embed_text.py
├── evaluate_retrieval.py
├── extract_pdf.py
├── search.py
├── test_answer_pipeline.py
├── test_gemini.py
├── data/
├── output/
├── .env
├── .gitignore
└── README.md
```

**Note:** The `data/`, `output/`, `.env`, and virtual environment entries represent local resources. They are not intended to be uploaded to GitHub.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/khadijahIrfan/DocuMind.git
cd DocuMind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the libraries required by the project. A `requirements.txt` file will be added after the dependencies are verified.

### 4. Configure environment variables

Create a local `.env` file and add the required environment variables, including `GEMINI_API_KEY` if your configured answer-generation pipeline uses Gemini.

Never commit API keys or other secrets.

### 5. Run the application

Review `app.py` for the application's entry point and required configuration before running it.

## Testing

Run the test suite with:

```bash
pytest
```

## Project Status

Under development. Setup instructions, dependency requirements, and usage examples will be refined as the project is prepared for reproducible use.

## Author

**Khadijah Irfan**

GitHub: [@khadijahIrfan](https://github.com/khadijahIrfan)
