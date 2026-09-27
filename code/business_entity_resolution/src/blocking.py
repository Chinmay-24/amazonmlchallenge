"""
Multi-strategy blocking/candidate generation for entity resolution.
Critical for scaling - reduces comparison space while maintaining high recall.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging
import re
from itertools import combinations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockingStrategy:
    """Multi-strategy blocking to generate candidate pairs."""
    
    def __init__(self, 
                 use_country_blocking: bool = True,
                 use_token_blocking: bool = True,
                 use_ngram_blocking: bool = True,
                 use_phonetic_blocking: bool = True,
                 ngram_size: int = 3,
                 min_token_length: int = 3):
        """
        Initialize blocking strategy.
        
        Args:
            use_country_blocking: Use exact country match
            use_token_blocking: Use token-based blocking
            use_ngram_blocking: Use character n-gram blocking
            use_phonetic_blocking: Use phonetic blocking
            ngram_size: Size of character n-grams
            min_token_length: Minimum token length for blocking
        """
        self.use_country_blocking = use_country_blocking
        self.use_token_blocking = use_token_blocking
        self.use_ngram_blocking = use_ngram_blocking
        self.use_phonetic_blocking = use_phonetic_blocking
        self.ngram_size = ngram_size
        self.min_token_length = min_token_length
    
    def generate_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams from text."""
        if not text:
            return set()
        
        text = text.lower().replace(" ", "")
        if len(text) < n:
            return {text}
        
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def generate_tokens(self, text: str, min_length: int = 3) -> Set[str]:
        """
        Generate meaningful tokens from text.
        
        Args:
            text: Input text
            min_length: Minimum token length
            
        Returns:
            Set of tokens
        """
        if not text:
            return set()
        
        # Split and filter
        tokens = text.lower().split()
        
        # Remove very short tokens and common stop words
        stop_words = {'the', 'and', 'or', 'of', 'in', 'at', 'to', 'for', 'a', 'an'}
        tokens = {t for t in tokens if len(t) >= min_length and t not in stop_words}
        
        return tokens
    
    def simple_phonetic(self, text: str) -> str:
        """
        Simple phonetic encoding (similar to Soundex but simpler).
        
        Args:
            text: Input text
            
        Returns:
            Phonetic code
        """
        if not text:
            return ""
        
        text = text.lower()
        
        # Remove common suffixes that don't affect pronunciation
        suffixes = ['ltd', 'limited', 'corp', 'corporation', 'inc', 'incorporated', 
                   'pvt', 'private', 'llc', 'llp', 'co', 'company']
        
        for suffix in suffixes:
            text = re.sub(r'\b' + suffix + r'\b', '', text)
        
        # Remove vowels except first character
        if text:
            first_char = text[0]
            rest = re.sub(r'[aeiou]', '', text[1:])
            text = first_char + rest
        
        # Remove consecutive duplicates
        result = []
        prev = None
        for char in text:
            if char != prev:
                result.append(char)
                prev = char
        
        return ''.join(result)[:6]  # Take first 6 characters
    
    def country_blocking(self, 
                        source1: pd.DataFrame, 
                        combined: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Block by exact country match.
        
        Args:
            source1: Source 1 dataframe
            combined: Combined source2+source3 dataframe
            
        Returns:
            Dictionary mapping s1_id -> set of candidate ids
        """
        candidates = defaultdict(set)
        
        # Group by country
        combined_by_country = combined.groupby('country_clean')['entity_id'].apply(set).to_dict()
        
        for _, s1_row in source1.iterrows():
            s1_id = s1_row['entity_id']
            country = s1_row['country_clean']
            
            # Get all entities from same country
            if country in combined_by_country:
                candidates[s1_id].update(combined_by_country[country])
        
        total_candidates = sum(len(v) for v in candidates.values())
        avg_candidates = total_candidates / len(source1) if len(source1) > 0 else 0
        
        logger.info(f"Country blocking: {total_candidates} total candidates, "
                   f"avg {avg_candidates:.1f} per S1 entity")
        
        return candidates
    
    def token_blocking(self, 
                      source1: pd.DataFrame, 
                      combined: pd.DataFrame,
                      field: str = 'business_name_clean') -> Dict[str, Set[str]]:
        """
        Block by shared tokens in business name or address.
        
        Args:
            source1: Source 1 dataframe
            combined: Combined source2+source3 dataframe
            field: Field to use for token blocking
            
        Returns:
            Dictionary mapping s1_id -> set of candidate ids
        """
        candidates = defaultdict(set)
        
        # Build inverted index: token -> set of entity_ids
        token_index = defaultdict(set)
        
        for _, row in combined.iterrows():
            tokens = self.generate_tokens(row[field], self.min_token_length)
            for token in tokens:
                token_index[token].add(row['entity_id'])
        
        # Find candidates for each source1 entity
        for _, s1_row in source1.iterrows():
            s1_id = s1_row['entity_id']
            s1_tokens = self.generate_tokens(s1_row[field], self.min_token_length)
            
            # Find entities sharing at least one token
            for token in s1_tokens:
                if token in token_index:
                    candidates[s1_id].update(token_index[token])
        
        total_candidates = sum(len(v) for v in candidates.values())
        avg_candidates = total_candidates / len(source1) if len(source1) > 0 else 0
        
        logger.info(f"Token blocking ({field}): {total_candidates} total candidates, "
                   f"avg {avg_candidates:.1f} per S1 entity")
        
        return candidates
    
    def ngram_blocking(self, 
                      source1: pd.DataFrame, 
                      combined: pd.DataFrame,
                      field: str = 'business_name_clean') -> Dict[str, Set[str]]:
        """
        Block by shared character n-grams.
        
        Args:
            source1: Source 1 dataframe
            combined: Combined source2+source3 dataframe
            field: Field to use for n-gram blocking
            
        Returns:
            Dictionary mapping s1_id -> set of candidate ids
        """
        candidates = defaultdict(set)
        
        # Build inverted index: ngram -> set of entity_ids
        ngram_index = defaultdict(set)
        
        for _, row in combined.iterrows():
            ngrams = self.generate_ngrams(row[field], self.ngram_size)
            for ngram in ngrams:
                ngram_index[ngram].add(row['entity_id'])
        
        # Find candidates for each source1 entity
        for _, s1_row in source1.iterrows():
            s1_id = s1_row['entity_id']
            s1_ngrams = self.generate_ngrams(s1_row[field], self.ngram_size)
            
            # Find entities sharing at least one n-gram
            for ngram in s1_ngrams:
                if ngram in ngram_index:
                    candidates[s1_id].update(ngram_index[ngram])
        
        total_candidates = sum(len(v) for v in candidates.values())
        avg_candidates = total_candidates / len(source1) if len(source1) > 0 else 0
        
        logger.info(f"N-gram blocking ({field}): {total_candidates} total candidates, "
                   f"avg {avg_candidates:.1f} per S1 entity")
        
        return candidates
    
    def phonetic_blocking(self, 
                         source1: pd.DataFrame, 
                         combined: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Block by phonetic similarity of business names.
        
        Args:
            source1: Source 1 dataframe
            combined: Combined source2+source3 dataframe
            
        Returns:
            Dictionary mapping s1_id -> set of candidate ids
        """
        candidates = defaultdict(set)
        
        # Build inverted index: phonetic_code -> set of entity_ids
        phonetic_index = defaultdict(set)
        
        for _, row in combined.iterrows():
            code = self.simple_phonetic(row['business_name_clean'])
            if code:
                phonetic_index[code].add(row['entity_id'])
        
        # Find candidates for each source1 entity
        for _, s1_row in source1.iterrows():
            s1_id = s1_row['entity_id']
            s1_code = self.simple_phonetic(s1_row['business_name_clean'])
            
            if s1_code in phonetic_index:
                candidates[s1_id].update(phonetic_index[s1_code])
        
        total_candidates = sum(len(v) for v in candidates.values())
        avg_candidates = total_candidates / len(source1) if len(source1) > 0 else 0
        
        logger.info(f"Phonetic blocking: {total_candidates} total candidates, "
                   f"avg {avg_candidates:.1f} per S1 entity")
        
        return candidates
    
    def generate_candidates(self, 
                          source1: pd.DataFrame, 
                          source2: pd.DataFrame,
                          source3: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generate candidate pairs using multi-strategy blocking.
        
        Args:
            source1: Source 1 dataframe (preprocessed)
            source2: Source 2 dataframe (preprocessed)
            source3: Source 3 dataframe (preprocessed)
            
        Returns:
            Dictionary mapping s1_id -> list of candidate ids
        """
        # Combine source2 and source3
        combined = pd.concat([source2, source3], ignore_index=True)
        
        logger.info(f"Generating candidates for {len(source1)} S1 entities "
                   f"against {len(combined)} S2+S3 entities")
        
        # Initialize with empty sets
        all_candidates = {row['entity_id']: set() for _, row in source1.iterrows()}
        
        # Strategy 1: Country blocking (if enabled)
        if self.use_country_blocking:
            country_candidates = self.country_blocking(source1, combined)
            for s1_id, cands in country_candidates.items():
                all_candidates[s1_id].update(cands)
        
        # Strategy 2: Token blocking on business name
        if self.use_token_blocking:
            token_name_candidates = self.token_blocking(source1, combined, 'business_name_clean')
            for s1_id, cands in token_name_candidates.items():
                all_candidates[s1_id].update(cands)
            
            # Token blocking on address
            token_addr_candidates = self.token_blocking(source1, combined, 'business_address_clean')
            for s1_id, cands in token_addr_candidates.items():
                all_candidates[s1_id].update(cands)
        
        # Strategy 3: N-gram blocking on business name
        if self.use_ngram_blocking:
            ngram_candidates = self.ngram_blocking(source1, combined, 'business_name_clean')
            for s1_id, cands in ngram_candidates.items():
                all_candidates[s1_id].update(cands)
        
        # Strategy 4: Phonetic blocking
        if self.use_phonetic_blocking:
            phonetic_candidates = self.phonetic_blocking(source1, combined)
            for s1_id, cands in phonetic_candidates.items():
                all_candidates[s1_id].update(cands)
        
        # Convert sets to sorted lists
        final_candidates = {k: sorted(list(v)) for k, v in all_candidates.items()}
        
        # Statistics
        total_candidates = sum(len(v) for v in final_candidates.values())
        avg_candidates = total_candidates / len(source1) if len(source1) > 0 else 0
        max_candidates = max(len(v) for v in final_candidates.values()) if final_candidates else 0
        entities_with_candidates = sum(1 for v in final_candidates.values() if v)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Final candidate generation statistics:")
        logger.info(f"  Total S1 entities: {len(source1)}")
        logger.info(f"  Entities with candidates: {entities_with_candidates}")
        logger.info(f"  Total candidate pairs: {total_candidates}")
        logger.info(f"  Average candidates per S1: {avg_candidates:.1f}")
        logger.info(f"  Max candidates for any S1: {max_candidates}")
        logger.info(f"  Reduction ratio: {len(combined) / avg_candidates:.1f}x" if avg_candidates > 0 else "N/A")
        logger.info(f"{'='*60}\n")
        
        return final_candidates


def main():
    """Test blocking strategies."""
    from data_loader import DataLoader
    
    loader = DataLoader()
    
    # Load and preprocess data
    s1_train, s2_train, s3_train = loader.load_source_data("train")
    s1_train = loader.preprocess_dataframe(s1_train)
    s2_train = loader.preprocess_dataframe(s2_train)
    s3_train = loader.preprocess_dataframe(s3_train)
    
    # Test blocking
    blocker = BlockingStrategy(
        use_country_blocking=True,
        use_token_blocking=True,
        use_ngram_blocking=True,
        use_phonetic_blocking=True
    )
    
    candidates = blocker.generate_candidates(s1_train, s2_train, s3_train)
    
    print(f"\nSample candidates:")
    for i, (s1_id, cands) in enumerate(list(candidates.items())[:5]):
        print(f"{s1_id}: {len(cands)} candidates")
        if cands:
            print(f"  {cands[:5]}")
        if i >= 4:
            break


if __name__ == "__main__":
    main()
