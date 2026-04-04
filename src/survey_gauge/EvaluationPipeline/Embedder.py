from typing import Literal
import hnswlib
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull


class Embedder:
    """Simple vector database that embeds texts and stores them."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_elements: int = 10000, 
                similarity_metric: Literal['l2', 'ip', 'cosine'] = 'cosine', ef_construction: int = 200, M: int = 16):
        """
        Initialize the Embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            max_elements: Maximum number of elements to store in the HNSW index
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize HNSW index for similarity search
        self.index = hnswlib.Index(space=similarity_metric, dim=self.embedding_dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        
        # Store the actual texts and their embeddings
        self.texts = []
        self.embeddings = np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        self.element_count = 0
    
    def add_texts(self, texts: list[str]) -> None:
        """
        Embed texts and add them to the vector database.
        
        Args:
            texts: List of text strings to embed and store
        """
        # Generate embeddings for the texts
        new_embeddings = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)
        
        # Add to HNSW index
        ids = np.arange(self.element_count, self.element_count + len(texts))
        self.index.add_items(new_embeddings, ids)
        
        # Store texts and embeddings
        self.texts.extend(texts)
        self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings.size > 0 else new_embeddings
        self.element_count += len(texts)
    
    def search(self, query_text: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Search for similar texts in the database.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            
        Returns:
            List of tuples (text, similarity_score)
        """
        query_embedding = self.model.encode(query_text, convert_to_numpy=True).astype(np.float32)
        labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=min(k, self.element_count))
        
        results = []
        for idx, distance in zip(labels[0], distances[0]):
            if idx < len(self.texts):
                # Convert cosine distance to similarity (1 - distance for cosine)
                similarity = 1 - distance
                results.append((self.texts[idx], similarity))
        return results
    
    def get_similarity_groups(self, distance_threshold: float = 0.5, min_group_size: int = 1) -> list[list[tuple[str, int]]]:
        """
        Group similar vectors in the database using hierarchical clustering.
        
        Args:
            distance_threshold: Distance threshold for grouping (lower = tighter clusters)
            min_group_size: Minimum number of items per group to include in results
            
        Returns:
            List of groups, where each group is a list of tuples (text, original_index)
        """
        if self.element_count == 0:
            print("No embeddings to group. Add texts first using add_texts().")
            return []
        
        if self.element_count == 1:
            return [[(self.texts[0], 0)]]
        
        # Compute pairwise distances
        distances = pdist(self.embeddings, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Perform hierarchical clustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='precomputed'
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Group texts by cluster
        groups = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in groups:
                groups[cluster_id] = []
            groups[cluster_id].append((self.texts[idx], idx))
        
        # Filter by minimum group size and sort by group size
        filtered_groups = [group for group in groups.values() if len(group) >= min_group_size]
        filtered_groups.sort(key=len, reverse=True)
        
        return filtered_groups
    
    def print_similarity_groups(self, distance_threshold: float = 0.5, min_group_size: int = 1) -> None:
        """
        Print grouped similar vectors with their texts in a readable format.
        
        Args:
            distance_threshold: Distance threshold for grouping (lower = tighter clusters)
            min_group_size: Minimum number of items per group to include in output
        """
        groups = self.get_similarity_groups(distance_threshold, min_group_size)
        
        if not groups:
            print("No groups found with the given parameters.")
            return
        
        print(f"\nFound {len(groups)} similarity group(s):\n")
        for group_idx, group in enumerate(groups, 1):
            print(f"Group {group_idx} ({len(group)} items):")
            print("-" * 60)
            for text, original_idx in group:
                print(f"  [{original_idx}] {text}")
            print()
    
    def _create_2d_projection(self):
        """
        Helper method to create 2D PCA projection of embeddings.
        
        Returns:
            Tuple of (pca_model, embeddings_2d)
        """
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)
        return pca, embeddings_2d
    
    def plot_similarity_groups(self, distance_threshold: float = 0.5, min_group_size: int = 1,
                               title: str = "Similarity Groups (2D Projection)", 
                               figsize: tuple = (14, 10), label_texts: bool = False) -> None:
        """
        Visualize similar groups in 2D using PCA, with each group colored differently and bordered.
        This method wraps plot_2d functionality and adds group boundary visualization.
        
        Args:
            distance_threshold: Distance threshold for grouping (lower = tighter clusters)
            min_group_size: Minimum number of items per group to include
            title: Title for the plot
            figsize: Figure size as (width, height)
            label_texts: Whether to label points with text content or indices
        """
        if self.element_count == 0:
            print("No embeddings to plot. Add texts first using add_texts().")
            return
        
        groups = self.get_similarity_groups(distance_threshold, min_group_size)
        
        if not groups:
            print("No groups found with the given parameters.")
            return
        
        # Create 2D projection
        pca, embeddings_2d = self._create_2d_projection()
        
        # Create color map for groups
        colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot each group with a different color
        for group_idx, group in enumerate(groups):
            indices = [original_idx for _, original_idx in group]
            x = embeddings_2d[indices, 0]
            y = embeddings_2d[indices, 1]
            plt.scatter(x, y, alpha=0.7, s=150, c=[colors[group_idx]], 
                       label=f"Group {group_idx + 1} ({len(group)} items)", edgecolors='black', linewidth=1)
            
            # Draw group boundaries (convex hull) if group has 3+ points
            if len(indices) >= 3:
                try:
                    points = embeddings_2d[indices]
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    # Close the hull by adding the first point at the end
                    hull_closed = np.vstack([hull_points, hull_points[0]])
                    plt.plot(hull_closed[:, 0], hull_closed[:, 1], 
                            color=colors[group_idx], linewidth=2, linestyle='--', alpha=0.6)
                except:
                    pass
            
            # Add labels if requested
            if label_texts:
                for text, original_idx in group:
                    label = text[:25] + "..." if len(text) > 25 else text
                    plt.annotate(label, (embeddings_2d[original_idx, 0], embeddings_2d[original_idx, 1]), 
                               fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')
            else:
                for _, original_idx in group:
                    plt.annotate(str(original_idx), (embeddings_2d[original_idx, 0], embeddings_2d[original_idx, 1]), 
                               fontsize=8, alpha=0.9, xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show()
    
    def plot_2d(self, title: str = "Vector Database Embeddings (2D Projection)", 
                figsize: tuple = (12, 8), label_texts: bool = False) -> None:
        """
        Plot the embeddings in 2D using PCA dimensionality reduction.
        
        Args:
            title: Title for the plot
            figsize: Figure size as (width, height)
            label_texts: Whether to label points with their text content
        """
        if self.element_count == 0:
            print("No embeddings to plot. Add texts first using add_texts().")
            return
        
        # Reduce dimensionality to 2D using PCA
        pca, embeddings_2d = self._create_2d_projection()
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=100, c=range(self.element_count))
        
        # Add labels if requested
        if label_texts:
            for i, text in enumerate(self.texts):
                # Truncate long texts for better visualization
                label = text[:30] + "..." if len(text) > 30 else text
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        plt.title(title)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.tight_layout()
        plt.show()
