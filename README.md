# E-commerce Consumer Behavior Analysis Pipeline

## Overview

This project provides a data analysis pipeline for processing e-commerce consumer behavior data stored in a CSV file. The pipeline preprocesses the data, generates structured documents for Table Retrieval-Augmented Generation (RAG), and verifies document quality. The documents are designed to be used with a LangChain FAISS vector store for advanced querying and analysis, enabling insights into customer demographics, purchase patterns, and behavior.

---

## Use Case

The primary use case is to analyze e-commerce consumer behavior data to uncover actionable insights, such as:

- **Customer Segmentation:** Understand purchasing patterns across demographics (e.g., gender, age, income level), purchase channels (e.g., online vs. in-store), and product categories.
- **Behavioral Analysis:** Identify trends in customer loyalty, discount usage, satisfaction, and engagement with ads.
- **Business Decision Support:** Provide data-driven insights for marketing strategies, product recommendations, and customer retention programs.
- **Query Capabilities:** Enable natural language queries (via a vector store) to answer complex questions, e.g.,  
  _"What percentage of unmarried customers who shop online purchase electronics?"_

The pipeline generates three types of documents:
- **Row-level Documents:** Detailed customer profiles with demographics, purchase details, and behavior.
- **Single-dimension Segment Documents:** Statistics for segments like gender, income level, or purchase category.
- **Multi-dimension Segment Documents:** Statistics for combinations of dimensions, e.g., gender + purchase channel.

These documents can be embedded in a FAISS vector store for efficient retrieval and analysis using large language models (LLMs).

---

## Requirements

### Software

- **Python:** Version 3.8 or higher
- **Dependencies:** Listed in `requirements.txt`
    - `pandas`
    - `numpy`
    - `langchain`
    - `IPython`

### Hardware

- **Memory:** At least 8GB RAM (for processing large CSV files)
- **Storage:** Sufficient space for the input CSV and output JSON files
- **CPU:** Multi-core processor recommended for faster data processing

### Input Data

A CSV file named `Ecommerce_Consumer_Behavior_Analysis_Data.csv` with the following columns:

`
Customer_ID, Age, Gender, Income_Level, Marital_Status, Education_Level, Occupation, Location,
Purchase_Category, Purchase_Amount, Frequency_of_Purchase, Purchase_Channel, Time_of_Purchase,
Brand_Loyalty, Product_Rating, Time_Spent_on_Product_Research(hours), Social_Media_Influence, Discount_Sensitivity, Return_Rate, Customer_Satisfaction, Engagement_with_Ads, Discount_Used, Customer_Loyalty_Program_Member, Purchase_Intent, Shipping_Preference, Time_to_Decision,
Device_Used_for_Shopping, Payment_Method
`


---

## Installation

### 1. Clone the Repository (or create the directory structure):

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


### 3. Install Dependencies:

```
pip install -r requirements.txt
```


### 4. Prepare the Input Data:

Place the `Ecommerce_Consumer_Behavior_Analysis_Data.csv` file in the project root directory  
**OR**  
Update the `CSV_PATH` in `config/settings.py` to point to its location.

---

## Implementation

### Run the Pipeline

Execute the main script to process the CSV, generate documents, and verify them:

```
python analyze_data.py
```



#### Expected Output

- **Console Output:** Information about the dataset (rows, columns, data types), preprocessing results, document creation progress, and verification results (e.g., document distribution, query capabilities).
- **JSON File:** A file named `table_rag_sample_documents.json` containing a sample of generated documents (customer rows, single-dimension segments, and multi-dimension segments).
- **Documents:** A list of LangChain `Document` objects ready for embedding in a FAISS vector store (not implemented in this script).

---

### Extending to Vector Store (Optional)

To use the generated documents with a LangChain FAISS vector store:

1. **Install additional dependencies:**
    ```
    pip install faiss-cpu sentence-transformers
    ```
    (or `faiss-gpu` for GPU support)

2. **Add the following code to `analyze_data.py` after document creation:**

    ```
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    ```

This creates a FAISS index that can be queried using LangChain for advanced analytics.

---

## Usage Example

Running `python analyze_data.py` will:

- Load and preprocess the CSV file, cleaning string columns, converting `Purchase_Amount` to numeric, and `Time_of_Purchase` to datetime.
- Generate documents:
    - Row-level documents for each customer.
    - Single-dimension segments (e.g., by gender, income level).
    - Multi-dimension segments (e.g., gender + purchase channel).
- Verify document quality, checking distribution and query capabilities for questions like:
    - "How many female customers used discounts?"
    - "What percentage of electronics purchases were made online?"
- Save a sample of documents to `table_rag_sample_documents.json`.

To query the data using a vector store, load the FAISS index and use LangChain's retrieval capabilities with an LLM.

---

## Troubleshooting

- **File Not Found Error:** Ensure the CSV file is in the correct path or update `CSV_PATH` in `config/settings.py`.
- **Memory Issues:** For large datasets, increase available RAM or process the CSV in chunks by modifying `data_preprocessing.py`.
- **Dependency Errors:** Verify all dependencies are installed correctly using `pip install -r requirements.txt`.
- **Data Format Issues:** Ensure the CSV columns match the expected schema. Missing or malformed columns may cause preprocessing errors.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows the existing style and includes tests for new features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


├── requirements.txt # Python dependencies
├── table_rag_sample_documents.json # Output file (generated)
└── README.md # This file



