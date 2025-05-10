from langchain.schema import Document
from config.settings import SINGLE_DIMENSIONS, AGE_GROUPS, MULTI_DIMENSIONS
from src.utils import format_value
import pandas as pd

def create_table_rag_documents_multidim(df):
    """Create documents for Table RAG from the e-commerce dataset."""
    documents = []

    # 1. Create row-level documents
    print("Creating row-level documents...")
    for idx, row in df.iterrows():
        documents.append(create_row_document(idx, row, df.columns))

    # 2. Create single-dimension segment statistics
    print("\nCreating single-dimension segment statistics...")
    segment_count = create_single_dimension_documents(df, documents)

    # 3. Create multi-dimension segment statistics
    print("\nCreating multi-dimension segment statistics...")
    multi_segment_count = create_multi_dimension_documents(df, documents)

    print(f"Created {segment_count} single-dimension segment documents")
    print(f"Created {multi_segment_count} multi-dimension segment documents")
    print(f"Total documents created: {len(documents)}")
    return documents

def create_row_document(idx, row, columns):
    """Create a row-level document for a customer."""
    content_parts = [f"Customer data (Row {idx}):"]

    # Demographics
    demographics = [
        f"Customer ID: {row['Customer_ID']}",
        f"Age: {row['Age']}",
        f"Gender: {row['Gender']}",
        f"Income Level: {row['Income_Level']}",
        f"Marital Status: {row['Marital_Status']}",
        f"Education: {row['Education_Level']}",
        f"Occupation Level: {row['Occupation']}",
        f"Location: {row['Location']}",
    ]
    content_parts.append("Demographics: " + " | ".join(demographics))

    # Purchase information
    purchase = [
        f"Category: {row['Purchase_Category']}",
        f"Amount: ${row['Purchase_Amount']:.2f}",
        f"Frequency: {row['Frequency_of_Purchase']} times",
        f"Channel: {row['Purchase_Channel']}",
        f"Date: {format_value(row['Time_of_Purchase'])}",
    ]
    content_parts.append("Purchase: " + " | ".join(purchase))

    # Customer behavior
    behavior = [
        f"Brand Loyalty: {row['Brand_Loyalty']}/5",
        f"Product Rating: {row['Product_Rating']}/5",
        f"Research Time: {row['Time_Spent_on_Product_Research(hours)']} hours",
        f"Social Media Influence: {row['Social_Media_Influence']}",
        f"Discount Sensitivity: {row['Discount_Sensitivity']}",
        f"Return Rate: {row['Return_Rate']}",
        f"Satisfaction: {row['Customer_Satisfaction']}/10",
        f"Ad Engagement: {row['Engagement_with_Ads']}",
        f"Used Discount: {row['Discount_Used']}",
        f"Loyalty Program Member: {row['Customer_Loyalty_Program_Member']}",
        f"Purchase Intent: {row['Purchase_Intent']}",
        f"Shipping Preference: {row['Shipping_Preference']}",
        f"Time to Decision: {row['Time_to_Decision']} days",
    ]
    content_parts.append("Shopping Behavior: " + " | ".join(behavior))

    # Device and payment
    tech = [
        f"Device: {row['Device_Used_for_Shopping']}",
        f"Payment: {row['Payment_Method']}",
    ]
    content_parts.append("Technology: " + " | ".join(tech))

    content = "\n".join(content_parts)
    metadata = {"doc_type": "customer_row", "row_idx": str(idx)}
    for col in columns:
        metadata[col] = format_value(row[col])

    return Document(page_content=content, metadata=metadata)

def create_single_dimension_documents(df, documents):
    """Create single-dimension segment statistics documents."""
    segment_count = 0

    for dim in SINGLE_DIMENSIONS:
        col, name = dim["column"], dim["name"]
        for value in df[col].unique():
            segment_data = df[df[col] == value]
            if len(segment_data) == 0:
                continue

            stats = calculate_segment_stats(segment_data, df)
            segment_title = f"{name}: {value}"
            content = create_segment_content(segment_title, stats, segment_data, col)
            metadata = create_segment_metadata(segment_title, col, value, stats)

            documents.append(Document(page_content=content, metadata=metadata))
            segment_count += 1

    # Process age groups
    for group in AGE_GROUPS:
        segment_data = df[
            (df["Age"] >= group["min"]) & (df["Age"] <= group["max"])
        ] if group["min"] != 0 else df[df["Age"] <= group["max"]]
        if len(segment_data) == 0:
            continue

        stats = calculate_segment_stats(segment_data, df)
        segment_title = f"Age Group: {group['label']}"
        content = create_segment_content(segment_title, stats, segment_data, "Age_Group")
        metadata = create_segment_metadata(segment_title, "Age_Group", group["label"], stats)

        documents.append(Document(page_content=content, metadata=metadata))
        segment_count += 1

    return segment_count

def create_multi_dimension_documents(df, documents):
    """Create multi-dimension segment statistics documents."""
    multi_segment_count = 0

    for dim_combo in MULTI_DIMENSIONS:
        dim1, dim2 = dim_combo["dim1"], dim_combo["dim2"]
        name1, name2 = dim_combo["name1"], dim_combo["name2"]

        values1 = [g["label"] for g in AGE_GROUPS] if dim1 == "Age_Group" else df[dim1].unique()
        values2 = df[dim2].unique()

        for val1 in values1:
            filtered_by_dim1 = filter_by_dimension(df, dim1, val1)
            if len(filtered_by_dim1) == 0:
                continue

            parent_stats = {
                "count": len(filtered_by_dim1),
                "total_count": len(df),
                "percentage": len(filtered_by_dim1) / len(df) * 100,
            }

            for val2 in values2:
                filtered_data = filtered_by_dim1[filtered_by_dim1[dim2] == val2]
                if len(filtered_data) == 0:
                    continue

                stats = calculate_segment_stats(filtered_data, df, filtered_by_dim1)
                segment_title = f"{name1}: {val1} + {name2}: {val2}"
                content = create_multi_segment_content(
                    segment_title, stats, filtered_data, dim1, dim2, filtered_by_dim1
                )
                metadata = create_multi_segment_metadata(segment_title, dim1, dim2, val1, val2, stats)

                documents.append(Document(page_content=content, metadata=metadata))
                multi_segment_count += 1

    return multi_segment_count

def calculate_segment_stats(segment_data, df, parent_data=None):
    """Calculate statistics for a segment."""
    stats = {
        "count": len(segment_data),
        "total_count": len(df),
        "percentage": len(segment_data) / len(df) * 100,
        "avg_purchase": segment_data["Purchase_Amount"].mean(),
        "total_purchase": segment_data["Purchase_Amount"].sum(),
        "avg_satisfaction": segment_data["Customer_Satisfaction"].mean(),
        "discount_usage": segment_data["Discount_Used"].mean() * 100,
        "loyalty_membership": segment_data["Customer_Loyalty_Program_Member"].mean() * 100,
    }
    if parent_data is not None:
        stats["parent_count"] = len(parent_data)
        stats["percentage_of_parent"] = len(segment_data) / len(parent_data) * 100
        stats["percentage_of_total"] = len(segment_data) / len(df) * 100
    return stats

def create_segment_content(segment_title, stats, segment_data, dim):
    """Create content for a single-dimension segment document."""
    content_parts = [
        f"Segment Analysis: {segment_title}",
        f"Total customers in this segment: {stats['count']} ({stats['percentage']:.1f}% of all customers)",
        f"Purchase metrics:",
        f"- Average purchase amount: ${stats['avg_purchase']:.2f}",
        f"- Total purchase amount: ${stats['total_purchase']:.2f}",
        f"- Average customer satisfaction: {stats['avg_satisfaction']:.1f}/10",
        f"Customer profile:",
        f"- Discount usage rate: {stats['discount_usage']:.1f}%",
        f"- Loyalty program membership: {stats['loyalty_membership']:.1f}%",
    ]

    if dim != "Purchase_Channel":
        channel_dist = segment_data["Purchase_Channel"].value_counts(normalize=True) * 100
        content_parts.append("Purchase channel distribution:")
        for channel, pct in channel_dist.items():
            content_parts.append(f"- {channel}: {pct:.1f}%")

    if dim != "Purchase_Category":
        top_categories = segment_data["Purchase_Category"].value_counts(normalize=True).head(5) * 100
        content_parts.append("Top product categories:")
        for category, pct in top_categories.items():
            content_parts.append(f"- {category}: {pct:.1f}%")

    if dim != "Device_Used_for_Shopping":
        device_dist = segment_data["Device_Used_for_Shopping"].value_counts(normalize=True) * 100
        content_parts.append("Device usage:")
        for device, pct in device_dist.items():
            content_parts.append(f"- {device}: {pct:.1f}%")

    return "\n".join(content_parts)

def create_multi_segment_content(segment_title, stats, filtered_data, dim1, dim2, parent_data):
    """Create content for a multi-dimension segment document."""
    content_parts = [
        f"Multi-Dimension Segment Analysis: {segment_title}",
        f"Customer counts:",
        f"- Total in this segment: {stats['count']}",
        f"- Percentage of {stats['count'] / stats['parent_count'] * 100:.1f}%",
        f"- Percentage of all customers: {stats['percentage_of_total']:.1f}%",
        f"Purchase metrics:",
        f"- Average purchase amount: ${stats['avg_purchase']:.2f}",
        f"- Total purchase amount: ${stats['total_purchase']:.2f}",
        f"- Average customer satisfaction: {stats['avg_satisfaction']:.1f}/10",
        f"Customer profile:",
        f"- Discount usage rate: {stats['discount_usage']:.1f}%",
        f"- Loyalty program membership: {stats['loyalty_membership']:.1f}%",
    ]

    if dim1 != "Gender" and dim2 != "Gender":
        gender_dist = filtered_data["Gender"].value_counts(normalize=True) * 100
        content_parts.append("Gender distribution:")
        for gender, pct in gender_dist.items():
            content_parts.append(f"- {gender}: {pct:.1f}%")

    if dim1 != "Purchase_Category" and dim2 != "Purchase_Category":
        top_categories = filtered_data["Purchase_Category"].value_counts(normalize=True).head(3) * 100
        content_parts.append("Top product categories:")
        for category, pct in top_categories.items():
            content_parts.append(f"- {category}: {pct:.1f}%")

    # Add insights
    if stats["percentage_of_parent"] > 50:
        content_parts.append(
            f"Insight: Most customers in this segment ({stats['percentage_of_parent']:.1f}%) are significant."
        )
    if stats["avg_purchase"] > parent_data["Purchase_Amount"].mean() * 1.2:
        content_parts.append(
            f"Insight: This segment spends significantly more than the average."
        )
    if stats["avg_satisfaction"] > parent_data["Customer_Satisfaction"].mean() * 1.1:
        content_parts.append(
            f"Insight: This segment has significantly higher satisfaction."
        )

    return "\n".join(content_parts)

def create_segment_metadata(segment_title, dim, value, stats):
    """Create metadata for a single-dimension segment."""
    return {
        "doc_type": "segment_statistics",
        "dimension": dim,
        "segment_value": str(value),
        "segment_name": segment_title,
        "count": str(stats["count"]),
        "percentage": f"{stats['percentage']:.1f}%",
        "avg_purchase": f"${stats['avg_purchase']:.2f}",
        "total_purchase": f"${stats['total_purchase']:.2f}",
        "avg_satisfaction": f"{stats['avg_satisfaction']:.1f}",
        "discount_usage": f"{stats['discount_usage']:.1f}%",
        "loyalty_membership": f"{stats['loyalty_membership']:.1f}%",
    }

def create_multi_segment_metadata(segment_title, dim1, dim2, val1, val2, stats):
    """Create metadata for a multi-dimension segment."""
    return {
        "doc_type": "multi_segment_statistics",
        "dimension1": dim1,
        "dimension2": dim2,
        "value1": str(val1),
        "value2": str(val2),
        "segment_name": segment_title,
        "count": str(stats["count"]),
        "parent_count": str(stats["parent_count"]),
        "percentage_of_parent": f"{stats['percentage_of_parent']:.1f}%",
        "percentage_of_total": f"{stats['percentage_of_total']:.1f}%",
        "avg_purchase": f"${stats['avg_purchase']:.2f}",
        "total_purchase": f"${stats['total_purchase']:.2f}",
        "avg_satisfaction": f"{stats['avg_satisfaction']:.1f}",
        "discount_usage": f"{stats['discount_usage']:.1f}%",
        "loyalty_membership": f"{stats['loyalty_membership']:.1f}%",
    }

def filter_by_dimension(df, dim, value):
    """Filter dataframe by dimension and value."""
    if dim == "Age_Group":
        group = next((g for g in AGE_GROUPS if g["label"] == value), None)
        if group:
            return (
                df[df["Age"] <= group["max"]]
                if group["min"] == 0
                else df[(df["Age"] >= group["min"]) & (df["Age"] <= group["max"])]
            )
        return pd.DataFrame()
    return df[df[dim] == value]
