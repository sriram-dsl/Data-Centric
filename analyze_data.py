from src.data_preprocessing import load_and_preprocess_data
from src.document_creation import create_table_rag_documents_multidim
from src.verification import verify_documents, check_query_capabilities, save_sample_documents
from src.vector_store import create_vector_store  # Add this import
from config.settings import CSV_PATH, EMBEDDING_MODEL, VECTOR_STORE_SAVE_PATH  # Update import

def main():
    # Load and preprocess data
    processed_df = load_and_preprocess_data(CSV_PATH)

    # Create documents
    documents = create_table_rag_documents_multidim(processed_df)

    # Create vector store
    create_vector_store(documents, EMBEDDING_MODEL, VECTOR_STORE_SAVE_PATH)  # Add this line

    # Verify documents
    verify_documents(documents)

    # Test query capabilities
    print("\nTesting document capabilities for analytical queries:")
    check_query_capabilities(
        documents,
        "Female customers who used discount",
        {
            "doc_type": "multi_segment_statistics",
            "dimension1": "Gender",
            "dimension2": "Discount_Used",
            "value1": "Female",
            "value2": "True",
        },
    )
    check_query_capabilities(
        documents,
        "Electronics products purchased online",
        {
            "doc_type": "multi_segment_statistics",
            "dimension1": "Purchase_Category",
            "dimension2": "Purchase_Channel",
            "value1": "Electronics",
            "value2": "Online",
        },
    )
    check_query_capabilities(
        documents,
        "Unmarried customers with online purchases",
        {
            "doc_type": "multi_segment_statistics",
            "dimension1": "Marital_Status",
            "dimension2": "Purchase_Channel",
            "value1": "Single",
            "value2": "Online",
        },
    )

    # Check specific example
    specific_example = check_query_capabilities(
        documents,
        "Unmarried people ordered electronics from online compared to total unmarried online orders",
        {
            "doc_type": "multi_segment_statistics",
            "dimension1": "Marital_Status",
            "dimension2": "Purchase_Channel",
            "value1": "Single",
            "value2": "Online",
        },
    )
    if specific_example:
        print("\nTo answer the specific example question:")
        print("1. Retrieve 'Single + Online' statistics")
        print("2. Retrieve 'Single + Electronics + Online' statistics")
        print("3. Calculate proportion")

    # Save sample documents
    save_sample_documents(documents)
    print("\nDocument verification complete!")

if __name__ == "__main__":
    main()
