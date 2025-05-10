import json
import random

def verify_documents(documents):
    """Verify the quality and distribution of created documents."""
    # Analyze document distribution
    doc_types = {}
    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    print("Document distribution by type:")
    for doc_type, count in doc_types.items():
        print(f"- {doc_type}: {count} documents")

    # Verify multi-dimension coverage
    if "multi_segment_statistics" in doc_types:
        dimension_combos = {}
        for doc in documents:
            if doc.metadata.get("doc_type") == "multi_segment_statistics":
                dim_pair = (doc.metadata.get("dimension1", ""), doc.metadata.get("dimension2", ""))
                dimension_combos[dim_pair] = dimension_combos.get(dim_pair, 0) + 1

        print("\nMulti-dimension coverage:")
        for dims, count in dimension_combos.items():
            print(f"- {dims[0]} + {dims[1]}: {count} segments")

    # Display sample multi-dimension segment
    multi_segment_docs = [doc for doc in documents if doc.metadata.get("doc_type") == "multi_segment_statistics"]
    if multi_segment_docs:
        print("\nSample multi-dimension segment document:")
        print(multi_segment_docs[0].page_content)
        print("\nMulti-segment metadata:")
        print(multi_segment_docs[0].metadata)

def find_matching_documents(documents, criteria, limit=3):
    """Find documents that match the specified criteria in metadata."""
    matches = []
    for doc in documents:
        if all(doc.metadata.get(key) == value for key, value in criteria.items()):
            matches.append(doc)
            if len(matches) >= limit:
                break
    return matches

def check_query_capabilities(documents, description, criteria):
    """Check if documents can answer a specific query."""
    matches = find_matching_documents(documents, criteria)
    print(f"\nQuery capability: {description}")
    print(f"Found {len(matches)} matching documents")

    if matches:
        print("Sample document:")
        print(f"- Type: {matches[0].metadata.get('doc_type')}")
        if "segment_name" in matches[0].metadata:
            print(f"- Segment: {matches[0].metadata.get('segment_name')}")
        print(f"- First 150 chars: {matches[0].page_content[:150]}...")

    return len(matches) > 0

def save_sample_documents(documents):
    """Save a sample of documents to JSON for inspection."""
    sample_docs = []
    for doc_type in ["customer_row", "segment_statistics", "multi_segment_statistics"]:
        docs = [doc for doc in documents if doc.metadata.get("doc_type") == doc_type]
        sample_docs.extend(random.sample(docs, min(5, len(docs))))

    sample_dict = [{"content": doc.page_content, "metadata": doc.metadata} for doc in sample_docs]
    with open("table_rag_sample_documents.json", "w") as f:
        json.dump(sample_dict, f, indent=2)

    print(f"\nSaved {len(sample_dict)} sample documents to 'table_rag_sample_documents.json'")
