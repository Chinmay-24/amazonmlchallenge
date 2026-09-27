"""
Data loading and preprocessing utilities for entity resolution.
Handles TSV files with proper tab separation.
"""

import pandas as pd
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess entity data from TSV files."""
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        
    def load_source_data(self, split: str = "train", nrows: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load source data files.
        
        Args:
            split: Either 'train' or 'test'
            nrows: Optional number of rows to load (for faster testing)
            
        Returns:
            Tuple of (source1_df, source2_df, source3_df)
        """
        base_path = os.path.join(self.data_dir, split)
        
        source1 = pd.read_csv(
            os.path.join(base_path, f"{split}_source1.tsv"), 
            sep="\t", 
            dtype=str,
            keep_default_na=False,
            nrows=nrows
        )
        source2 = pd.read_csv(
            os.path.join(base_path, f"{split}_source2.tsv"), 
            sep="\t", 
            dtype=str,
            keep_default_na=False,
            nrows=nrows
        )
        source3 = pd.read_csv(
            os.path.join(base_path, f"{split}_source3.tsv"), 
            sep="\t", 
            dtype=str,
            keep_default_na=False,
            nrows=nrows
        )
        
        logger.info(f"Loaded {split} data: S1={len(source1)}, S2={len(source2)}, S3={len(source3)}")
        
        return source1, source2, source3
    
    def load_ground_truth(self) -> pd.DataFrame:
        """Load training ground truth data."""
        gt_path = os.path.join(self.data_dir, "train", "train_ground_truth.tsv")
        
        gt = pd.read_csv(
            gt_path, 
            sep="\t", 
            dtype=str,
            keep_default_na=False
        )
        
        logger.info(f"Loaded ground truth: {len(gt)} records")
        
        return gt
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for matching.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, countries_filter: set = None) -> pd.DataFrame:
        """
        Preprocess a source dataframe.
        
        Args:
            df: Source dataframe
            countries_filter: Optional set of countries to filter by (for optimization)
            
        Returns:
            Preprocessed dataframe with additional columns
        """
        df = df.copy()
        
        # Early filtering by country if specified (huge speedup)
        if countries_filter is not None:
            initial_len = len(df)
            df['country_clean'] = df['country'].apply(self.preprocess_text)
            df = df[df['country_clean'].isin(countries_filter)]
            logger.info(f"Country filter: {initial_len} → {len(df)} records ({100*len(df)/initial_len:.1f}% kept)")
            
            # Create preprocessed versions (only for filtered records)
            df['business_name_clean'] = df['business_name'].apply(self.preprocess_text)
            df['business_address_clean'] = df['business_address'].apply(self.preprocess_text)
        else:
            # Create preprocessed versions
            df['business_name_clean'] = df['business_name'].apply(self.preprocess_text)
            df['business_address_clean'] = df['business_address'].apply(self.preprocess_text)
            df['country_clean'] = df['country'].apply(self.preprocess_text)
        
        # Extract tokens for blocking
        df['name_tokens'] = df['business_name_clean'].apply(
            lambda x: set(x.split()) if x else set()
        )
        df['address_tokens'] = df['business_address_clean'].apply(
            lambda x: set(x.split()) if x else set()
        )
        
        return df
    
    def parse_ground_truth(self, gt_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Parse ground truth into a dictionary.
        
        Args:
            gt_df: Ground truth dataframe
            
        Returns:
            Dictionary mapping source1_id -> list of matched ids
        """
        gt_dict = {}
        
        for _, row in gt_df.iterrows():
            s1_id = row['source1_entity_id']
            matched_ids = row['matched_entity_ids']
            
            if matched_ids and matched_ids.strip():
                gt_dict[s1_id] = [mid.strip() for mid in matched_ids.split(',')]
            else:
                gt_dict[s1_id] = []
        
        return gt_dict
    
    def combine_sources(self, source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
        """
        Combine source2 and source3 into a single dataframe for matching.
        
        Args:
            source2: Source 2 dataframe
            source3: Source 3 dataframe
            
        Returns:
            Combined dataframe
        """
        combined = pd.concat([source2, source3], ignore_index=True)
        logger.info(f"Combined sources: {len(combined)} records")
        
        return combined
    
    def create_train_pairs(self, 
                          source1: pd.DataFrame, 
                          combined: pd.DataFrame,
                          gt_dict: Dict[str, List[str]],
                          negative_ratio: float = 2.0) -> pd.DataFrame:
        """
        Create training pairs (positive and negative examples).
        
        Args:
            source1: Source 1 dataframe
            combined: Combined source2+source3 dataframe
            gt_dict: Ground truth dictionary
            negative_ratio: Ratio of negative to positive examples
            
        Returns:
            Dataframe with training pairs
        """
        import random
        
        pairs = []
        
        # Create a dict for fast lookup
        combined_dict = combined.set_index('entity_id').to_dict('index')
        
        for _, s1_row in source1.iterrows():
            s1_id = s1_row['entity_id']
            matched_ids = gt_dict.get(s1_id, [])
            
            # Positive examples
            for matched_id in matched_ids:
                if matched_id in combined_dict:
                    pairs.append({
                        's1_id': s1_id,
                        's2s3_id': matched_id,
                        'label': 1
                    })
            
            # Negative examples (random sampling)
            n_negatives = int(len(matched_ids) * negative_ratio) if matched_ids else 5
            
            # Sample from same country for harder negatives
            same_country = combined[combined['country'] == s1_row['country']]
            available_negatives = same_country[~same_country['entity_id'].isin(matched_ids)]
            
            if len(available_negatives) > 0:
                n_sample = min(n_negatives, len(available_negatives))
                negative_samples = available_negatives.sample(n=n_sample, random_state=42)
                
                for _, neg_row in negative_samples.iterrows():
                    pairs.append({
                        's1_id': s1_id,
                        's2s3_id': neg_row['entity_id'],
                        'label': 0
                    })
        
        pairs_df = pd.DataFrame(pairs)
        logger.info(f"Created {len(pairs_df)} training pairs: "
                   f"{sum(pairs_df['label']==1)} positive, {sum(pairs_df['label']==0)} negative")
        
        return pairs_df


def main():
    """Test data loading."""
    loader = DataLoader()
    
    # Load training data
    s1_train, s2_train, s3_train = loader.load_source_data("train")
    gt_train = loader.load_ground_truth()
    
    # Preprocess
    s1_train = loader.preprocess_dataframe(s1_train)
    s2_train = loader.preprocess_dataframe(s2_train)
    s3_train = loader.preprocess_dataframe(s3_train)
    
    print("\nSample Source 1 data:")
    print(s1_train.head())
    
    print("\nGround truth sample:")
    print(gt_train.head())
    
    # Parse ground truth
    gt_dict = loader.parse_ground_truth(gt_train)
    print(f"\nGround truth statistics:")
    print(f"Total S1 entities: {len(gt_dict)}")
    print(f"Entities with matches: {sum(1 for v in gt_dict.values() if v)}")
    print(f"Singletons: {sum(1 for v in gt_dict.values() if not v)}")


if __name__ == "__main__":
    main()
