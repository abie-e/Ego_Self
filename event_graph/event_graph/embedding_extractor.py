"""
Gemini Embedding Extractor
Provides text embedding extraction using Google Gemini embedding API via OpenAI SDK
"""

import os
import yaml
from openai import OpenAI
from typing import List, Optional
from pathlib import Path
import numpy as np


class EmbeddingExtractor:
    """Google Gemini embedding model for extracting embeddings from text"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Gemini Embedding client via OpenAI SDK

        Args:
            config_path: Path to config.yaml file. If None, looks in default location.
        """
        # Load configuration
        if config_path is None:
            # Default config path relative to this file
            config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        embedding_config = config['api']['embedding']
        self.model = embedding_config['model']
        self.embedding_dim = embedding_config['embedding_dim']

        # Initialize OpenAI client with Gemini base URL
        self.client = OpenAI(
            api_key=embedding_config['api_key'],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        print(f"Loaded Gemini Embedding API via OpenAI SDK: {self.model}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def extract_text_embedding(self, text: str) -> List[float]:
        """
        Extract embedding from text using OpenAI SDK

        Args:
            text: Input text string

        Returns:
            Embedding as list of floats (normalized)
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dim

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding

            # Normalize embedding
            embedding_np = np.array(embedding)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm

            return embedding_np.tolist()

        except Exception as e:
            print(f"Error extracting embedding for text '{text[:50]}...': {e}")
            return [0.0] * self.embedding_dim

    def extract_batch_text_embeddings(self, texts: List[str],
                                     batch_size: int = 32) -> List[List[float]]:
        """
        Extract embeddings from multiple texts in batches using OpenAI SDK

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            List of embeddings
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Filter out empty texts
            valid_texts = [t if t and t.strip() else " " for t in batch_texts]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=valid_texts
                )

                # Extract and normalize embeddings
                batch_embeddings = []
                for data in response.data:
                    embedding_np = np.array(data.embedding)
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        embedding_np = embedding_np / norm
                    batch_embeddings.append(embedding_np.tolist())

                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error extracting batch embeddings: {e}")
                # Return zero vectors for failed batch
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch_texts))

        return embeddings

    def cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)
        """
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)

        norm1 = np.linalg.norm(emb1_np)
        norm2 = np.linalg.norm(emb2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1_np, emb2_np) / (norm1 * norm2))

    def save_embedding(self, embedding: List[float], event_id: str,
                      output_dir: str = "data/features/embeddings") -> str:
        """
        Save embedding to .npy file

        Args:
            embedding: Embedding vector as list of floats
            event_id: Event ID (e.g., 'DAY1_001.evt')
            output_dir: Directory to save embeddings (relative or absolute path)

        Returns:
            Path to saved embedding file
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename: event_id + .npy
        filename = f"{event_id}.npy"
        filepath = output_path / filename

        # Save embedding as numpy array
        embedding_np = np.array(embedding, dtype=np.float32)
        np.save(filepath, embedding_np)

        return str(filepath)

    def load_embedding(self, event_id: str,
                      input_dir: str = "data/features/embeddings") -> Optional[List[float]]:
        """
        Load embedding from .npy file

        Args:
            event_id: Event ID (e.g., 'DAY1_001.evt')
            input_dir: Directory containing embeddings

        Returns:
            Embedding as list of floats, or None if file doesn't exist
        """
        input_path = Path(input_dir)
        filename = f"{event_id}.npy"
        filepath = input_path / filename

        if not filepath.exists():
            return None

        # Load embedding
        embedding_np = np.load(filepath)
        return embedding_np.tolist()


# Singleton instance for easy access
_extractor_instance = None

def get_extractor(config_path: Optional[str] = None) -> EmbeddingExtractor:
    """
    Get or create singleton EmbeddingExtractor instance

    Args:
        config_path: Path to config.yaml file

    Returns:
        EmbeddingExtractor instance
    """
    global _extractor_instance

    if _extractor_instance is None:
        _extractor_instance = EmbeddingExtractor(config_path)

    return _extractor_instance
