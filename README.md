# E-commerce Data Analysis and QA Pair Generation Pipeline

This guide provides instructions for setting up and running the integrated e-commerce consumer behavior analysis and QA pair generation pipeline. The pipeline processes e-commerce data, creates a FAISS vector store for Retrieval-Augmented Generation (RAG), and generates GSM8K-style question-answer (QA) pairs for fine-tuning large language models (LLMs) using Group Preference Optimization (GPO).

![image](https://github.com/user-attachments/assets/bbf0c4f1-0fc2-40f0-b297-882867bfffb1)


---

## Overview

The pipeline consists of two main components:

1. **Data Analysis Pipeline:**
    - Loads and preprocesses e-commerce consumer behavior data from a CSV file.
    - Generates structured documents (row-level, single-dimension, and multi-dimension segments).
    - Creates a FAISS vector store for RAG-based querying.
    - Verifies document quality and query capabilities.

2. **QA Pair Generation Pipeline:**
    - Generates data-driven analytical questions based on the vector store.
    - Answers questions with step-by-step reasoning in GSM8K format.
    - Formats and validates QA pairs for LLM fine-tuning with GPO.

---

## Use Case

This pipeline is designed for:

- **Business Insights:** Analyzing customer demographics, purchase patterns, and behavior to inform marketing and product strategies.
- **LLM Fine-Tuning:** Generating high-quality, mathematical QA pairs in GSM8K format to fine-tune LLMs for e-commerce analytics tasks.
- **Query Capabilities:** Enabling natural language queries to answer complex questions like "What percentage of unmarried customers who shop online purchase electronics?"

---

## Requirements

### Software

- **Python:** Version 3.8 or higher.
- **Ollama:** For running the llama3 LLM and nomic-embed-text embedding model locally.
- **Dependencies (listed in `requirements.txt`):**
    - pandas
    - numpy
    - langchain
    - IPython
    - langchain_ollama
    - faiss-cpu (or faiss-gpu for GPU support)

### Hardware

- **Memory:** At least 8GB RAM (16GB recommended for large datasets or QA generation).
- **Storage:** Sufficient space for the input CSV, FAISS index (~1-2GB), and QA output files.
- **CPU/GPU:** Multi-core CPU or GPU (optional) for faster processing.

### Input Data

A CSV file named `Ecommerce_Consumer_Behavior_Analysis_Data.csv` with columns:

`
Customer_ID, Age, Gender, Income_Level, Marital_Status, Education_Level, Occupation, Location,
Purchase_Category, Purchase_Amount, Frequency_of_Purchase, Purchase_Channel, Time_of_Purchase,
Brand_Loyalty, Product_Rating, Time_Spent_on_Product_Research(hours), Social_Media_Influence, Discount_Sensitivity, Return_Rate, Customer_Satisfaction, Engagement_with_Ads, Discount_Used, Customer_Loyalty_Program_Member, Purchase_Intent, Shipping_Preference, Time_to_Decision,
Device_Used_for_Shopping, Payment_Method
`


---

## Project Structure

```
Data-Centric/
├── config/
│ └── settings.py # Configuration for dimensions, paths, and QA settings
├── src/
│ ├── data_preprocessing.py # Data loading and preprocessing
│ ├── document_creation.py # Document creation logic
│ ├── verification.py # Document verification and query testing
│ ├── utils.py # Helper functions
│ ├── vector_store.py # Vector store creation and saving
│ ├── qa/
│ │ ├── question_generator.py # Question generation logic
│ │ ├── answer_generator.py # Answer generation logic
│ │ ├── qa_formatter.py # QA pair formatting and validation
│ │ └── pipeline.py # QA pipeline orchestration
├── analyze_data.py # Main script to run the full pipeline
├── requirements.txt # Python dependencies
├── table_rag_sample_documents.json # Output file (generated)
├── qa_outputs/ # QA pipeline output directory (generated)
└── README.md # Project documentation
```


---

## Installation

### 1. Clone the Repository (or set up the directory structure):

```
git clone https://github.com/sriram-dsl/Data-Centric.git
cd Data-Centric
```

Alternatively, create the directory structure as shown above and save the provided files.

### 2. Set Up a Virtual Environment (recommended):

```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```


### 3. Install Python Dependencies:

```
pip install -r requirements.txt
```


### 4. Install Ollama:

- **Linux:**
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ```
- **macOS:** Download and install from [https://ollama.com/download](https://ollama.com/download).
- **Windows:** Use WSL2 (see Microsoft's WSL2 guide) and follow Linux instructions.

Verify installation:
```
ollama --version
```


### 5. Pull Ollama Models:

- Start the Ollama service:
    ```
    ollama serve &
    ```
- Pull the required models:
    ```
    ollama pull nomic-embed-text
    ollama pull llama3
    ```
- Verify models:
    ```
    ollama list
    ```

### 6. Prepare Input Data:

Place `Ecommerce_Consumer_Behavior_Analysis_Data.csv` in the project root or update `CSV_PATH` in `config/settings.py` to point to its location.

---

## Running the Pipeline

Execute the Pipeline:  
Run the main script to process the CSV, create the vector store, verify documents, and generate QA pairs:

```
python analyze_data.py
```


---

## Expected Output

### Console Output

- Data preprocessing details (rows, columns, data types).
- Document creation progress (row-level, single-dimension, multi-dimension).
- Vector store creation confirmation.
- Document verification results (distribution, query capabilities).
- QA pair generation progress (questions, answers, formatted pairs).

### Files

- `table_rag_sample_documents.json`: Sample of generated documents.
- `ecommerce_table_rag/`: FAISS vector store directory.
- `qa_outputs/`:
    - `formatted_qa_pairs_final.json`: All formatted QA pairs.
    - `gsm8k_formatted_qa_pairs.json`: QA pairs in strict GSM8K format.
    - Individual QA pair JSON files (e.g., `qa_pair_1.json`).

### Documents

- LangChain Document objects embedded in the FAISS vector store.

### QA Pairs

- GSM8K-style QA pairs for LLM fine-tuning.

### finetuning the LLM with GRPO 

- after running `analyze_data.py` you will get the `qa_outputs/gsm8k_formatted_qa_pairs.json`
- take that resulted json file and running finetuning notebook(`finetuning_llm.ipynb`) mentioned in the repo 
---

## Configuration

Adjust settings in `config/settings.py`:

- `CSV_PATH`: Path to the input CSV.
- `EMBEDDING_MODEL`: Ollama embedding model (`nomic-embed-text`).
- `VECTOR_STORE_SAVE_PATH`: FAISS index path (`ecommerce_table_rag`).
- `QA_LLM_MODEL`: LLM for QA generation (`llama3`).
- `QA_OUTPUT_DIR`: QA output directory (`qa_outputs`).
- `QA_NUM_QUESTIONS_PER_CATEGORY`: Questions per category (default: 5).
- `QA_TOTAL_QUESTIONS`: Total QA pairs to generate (default: 20).
- `QA_CATEGORIES`: List of query categories for QA generation.

---

## Usage Example

Running `python analyze_data.py` will:

- Preprocess the CSV, cleaning strings and converting data types.
- Generate documents (customer profiles, segment statistics).
- Create a FAISS vector store using nomic-embed-text embeddings.
- Verify document quality and test query capabilities (e.g., "How many female customers used discounts?").
- Generate 20 GSM8K-style QA pairs across 8 categories (e.g., customer demographics, discount usage).
- Save outputs to `table_rag_sample_documents.json` and `qa_outputs/`.

**To use the QA pairs for LLM fine-tuning:**

- Use `qa_outputs/gsm8k_formatted_qa_pairs.json` as input for GPO fine-tuning workflows.
- The pairs are formatted with questions (self-contained word problems) and answers (step-by-step reasoning with `<<calculation=result>>` and `#### [number]`).

---

## Troubleshooting

- **Ollama Errors:**
    - Ensure Ollama is running (`ollama serve &`) and models are pulled (`ollama list`).
    - Check for sufficient disk space and RAM for llama3 and nomic-embed-text.
- **File Not Found:**
    - Verify `Ecommerce_Consumer_Behavior_Analysis_Data.csv` is in the correct path.
    - Update `CSV_PATH` in `config/settings.py` if needed.
- **Dependency Issues:**
    - Run `pip install -r requirements.txt` to install all dependencies.
    - Upgrade pip if needed: `pip install --upgrade pip`.
- **Memory Issues:**
    - Increase RAM or process the CSV in chunks by modifying `data_preprocessing.py`.
    - Reduce `QA_TOTAL_QUESTIONS` in `config/settings.py` for smaller datasets.
- **QA Pair Validation Failures:**
    - Check console logs for validation errors (e.g., missing `<<calculation>>` or `####`).
    - Adjust `QA_NUM_QUESTIONS_PER_CATEGORY` or increase `max_attempts` in `pipeline.py` for more retries.
- **Data Format Issues:**
    - Ensure CSV columns match the expected schema.
    - Fix malformed data (e.g., inconsistent date formats) before running.

## note : any doubts in the dataset preparation step check the `Untitled.ipynb` notebook 
---

## Additional Resources

- [Ollama Documentation](https://ollama.com/docs): Setup and model management.
- [LangChain Documentation](https://python.langchain.com/docs/): FAISS and Ollama integration.
- [GSM8K Dataset](https://github.com/openai/grade-school-math): Reference for QA pair format.
- [FAISS Documentation](https://faiss.ai/): Vector store details.
