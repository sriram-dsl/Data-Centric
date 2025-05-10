# Setting Up Ollama and Embedding Model for E-commerce Data Analysis

This guide provides step-by-step instructions to set up Ollama and pull the `nomic-embed-text` embedding model, which is required for creating a FAISS vector store in the e-commerce consumer behavior analysis pipeline.

---

## Prerequisites

- **Operating System:** Linux, macOS, or Windows (with WSL2 recommended for Windows users)
- **Hardware:** At least 8GB RAM and sufficient disk space (the `nomic-embed-text` model requires approximately 1-2GB)
- **Python Environment:** Python 3.8 or higher with project dependencies installed (see `requirements.txt`)
- **Internet Connection:** Required to download Ollama and the embedding model

---

## Installation Steps

### 1. Install Ollama

Ollama is a tool for running large language models and embedding models locally. Follow these steps to install it:

#### Linux

1. Open a terminal.
2. Run the following command to download and install Ollama:
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ```
3. Verify the installation:
    ```
    ollama --version
    ```
   You should see the Ollama version number.

#### macOS

1. Download the Ollama installer from [https://ollama.com/download](https://ollama.com/download).
2. Run the installer and follow the prompts.
3. Open a terminal and verify the installation:
    ```
    ollama --version
    ```

#### Windows (via WSL2)

1. Set up WSL2 by following Microsoft's [WSL2 installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).
2. Open a WSL2 terminal (e.g., Ubuntu).
3. Install Ollama as described in the Linux section above.
4. Verify the installation:
    ```
    ollama --version
    ```

---

### 2. Start the Ollama Service

Ollama must be running to serve the embedding model:

1. Start the Ollama service in the background:
    ```
    ollama serve &
    ```
2. Verify that Ollama is running:
    ```
    ollama list
    ```
   This command lists available models (initially empty).

---

### 3. Pull the `nomic-embed-text` Model

The pipeline uses the `nomic-embed-text` model for generating embeddings:

1. Pull the model using the following command:
    ```
    ollama pull nomic-embed-text
    ```
   This downloads the model (approximately 1-2GB) and may take a few minutes depending on your internet speed.

2. Verify that the model is available:
    ```
    ollama list
    ```
   You should see `nomic-embed-text:latest` in the output.

---

## Next Steps

Once Ollama is running and the embedding model is available, you can proceed with the e-commerce consumer behavior analysis pipeline. The FAISS vector store setup will use the `nomic-embed-text` model for generating embeddings as required by your workflow.

---

