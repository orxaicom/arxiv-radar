import csv
import os
import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
import json
import ast


def read_csv(file_path):
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return data


def generate_umap_and_clusters(csv_file_path):
    # Read data from CSV
    csv_data = read_csv(csv_file_path)

    # Initialize lists to store embeddings, keys, and additional information
    embeddings = []
    keys = []
    additional_info = []

    # Create a dictionary to store data for each field
    field_data = {}

    # Retrieve embeddings and additional information from CSV data
    for row in csv_data:
        arxiv_id = row["arxiv"]
        embedding_bytes = ast.literal_eval(row["embedding"])
        title = row["title"]
        abstract = row["abstract"]
        field = row["field"]
        categories = row["categories"]

        # Check if all fields exist
        if embedding_bytes and title and abstract and field:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding.tolist())  # Convert to Python list
            keys.append(arxiv_id)
            additional_info.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "field": field,
                    "categories": categories,
                }
            )

            # Update field_data dictionary
            if field not in field_data:
                field_data[field] = {
                    "embeddings": [],
                    "keys": [],
                    "additional_info": [],
                }
            field_data[field]["embeddings"].append(embedding.tolist())
            field_data[field]["keys"].append(arxiv_id)
            field_data[field]["additional_info"].append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "field": field,
                    "categories": categories,
                }
            )

    # Check if there are any embeddings
    if not embeddings:
        print("No embeddings found. Exiting.")
        exit()

    # Convert lists to numpy arrays
    embeddings = np.array(embeddings)

    # Perform UMAP dimensionality reduction in 2D with adjusted parameters for the entire dataset
    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    embedded_embeddings = umap.fit_transform(embeddings)

    # Save the UMAP embeddings for the entire dataset to a JSON file
    umap_data = {
        "embeddings": embedded_embeddings.tolist(),
        "keys": keys,
        "additional_info": additional_info,
    }
    umap_output_path = os.path.join("output", "umap_data_All.json")
    with open(umap_output_path, "w") as json_file:
        json.dump(umap_data, json_file)

    # Perform K-Means clustering for the entire dataset
    num_clusters_all = min(len(embeddings), 5)  # Adjust the number of clusters as needed
    kmeans_all = KMeans(n_clusters=num_clusters_all, random_state=42)
    clusters_all = kmeans_all.fit_predict(embeddings)

    # Save the cluster information for the entire dataset to a JSON file
    cluster_data_all = {"clusters": clusters_all.tolist()}
    cluster_output_path = os.path.join("output", "cluster_data_All.json")
    with open(cluster_output_path, "w") as json_file:
        json.dump(cluster_data_all, json_file)

    # Generate UMAP and cluster data for each field
    for field, field_info in field_data.items():
        field_embeddings = np.array(field_info["embeddings"])

        # Perform UMAP dimensionality reduction for each field
        umap_field = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
        embedded_embeddings_field = umap_field.fit_transform(field_embeddings)

        # Save the UMAP embeddings for each field to a JSON file
        umap_data_field = {
            "embeddings": embedded_embeddings_field.tolist(),
            "keys": field_info["keys"],
            "additional_info": field_info["additional_info"],
        }
        umap_output_field_path = os.path.join(
            "output", f"umap_data_{field.replace(' ', '_')}.json"
        )
        with open(umap_output_field_path, "w") as json_file:
            json.dump(umap_data_field, json_file)

        # Perform K-Means clustering for each field
        num_clusters_field = min(len(field_embeddings), 5)  # Adjust the number of clusters as needed
        kmeans_field = KMeans(n_clusters=num_clusters_field, random_state=42)
        clusters_field = kmeans_field.fit_predict(field_embeddings)

        # Save the cluster information for each field to a JSON file
        cluster_data_field = {"clusters": clusters_field.tolist()}
        cluster_output_field_path = os.path.join(
            "output", f"cluster_data_{field.replace(' ', '_')}.json"
        )
        with open(cluster_output_field_path, "w") as json_file:
            json.dump(cluster_data_field, json_file)


if __name__ == "__main__":
    csv_file_path = "daily-arxiv-embeddings.csv"
    generate_umap_and_clusters(csv_file_path)
