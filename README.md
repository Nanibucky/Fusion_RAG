# RAG with Multi-Query Fusion

This project implements a Retrieval-Augmented Generation (RAG) system enhanced with Multi-Query Fusion for improved document retrieval and question answering. It uses OpenAI's language models and embeddings to process PDF documents and answer user queries.

## Features

- **Document Loading**: Loads and indexes PDF documents from a specified directory.  
- **Multi-Query Fusion**: Generates multiple query variations to improve retrieval accuracy.  
- **Reciprocal Rank Fusion (RRF)**: Combines results from multiple queries to rank the most relevant documents.  
- **Question Answering**: Answers user questions based on the retrieved document chunks.  

## Setup

### Prerequisites

- Python 3.8 or higher  
- OpenAI API key (add it to a `.env` file as `OPENAI_API_KEY=your_api_key`)  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your PDF documents to the `docs` directory (or specify a custom directory).

## Configuration

Update the `.env` file with your OpenAI API key and optional settings:

### `.env` file:
```ini
OPENAI_API_KEY=your_api_key
DOCS_DIR=./docs  # Path to your PDF documents
LLM_MODEL=gpt-4  # Default LLM model
EMBEDDING_MODEL=text-embedding-3-small  # Default embedding model
```

## Usage

Run the script:
```bash
python main.py
```

- If the documents directory is not specified in the `.env` file, you will be prompted to enter the path manually.  
- Ask questions interactively. Type `exit` to quit.  
