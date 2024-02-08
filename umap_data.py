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


def generate_random_data(size, n_components=2):
    # Generate random embeddings and random cluster assignments
    random_embeddings = np.random.rand(size, n_components)
    random_clusters = np.random.randint(0, 5, size)
    return random_embeddings, random_clusters


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
        field = row["field"]
        categories = row["categories"]

        # Check if all fields exist
        if embedding_bytes and title and field:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding.tolist())  # Convert to Python list
            keys.append(arxiv_id)
            additional_info.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
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
                    "field": field,
                    "categories": categories,
                }
            )

    # Ensure the output directory exists
    os.makedirs("output", exist_ok=True)

    # Generate UMAP and cluster data for each field
    for field, field_info in field_data.items():
        if len(field_info["embeddings"]) < 5:
            # If less than 5 embeddings, generate random data
            random_embeddings, random_clusters = generate_random_data(
                len(field_info["embeddings"])
            )
            embedded_embeddings_field = random_embeddings
            clusters_field = random_clusters
        else:
            # Perform UMAP dimensionality reduction for each field
            field_embeddings = np.array(field_info["embeddings"])
            umap_field = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
            embedded_embeddings_field = umap_field.fit_transform(field_embeddings)

            # Perform K-Means clustering for each field
            num_clusters_field = min(len(field_embeddings), 5)  # Adjust the number of clusters as needed
            kmeans_field = KMeans(n_clusters=num_clusters_field, random_state=42)
            clusters_field = kmeans_field.fit_predict(field_embeddings)

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
