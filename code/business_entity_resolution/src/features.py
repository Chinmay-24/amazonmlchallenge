"""
Feature engineering for entity matching.
Comprehensive string similarity and pattern-based features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate features for entity pair matching."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
    
    # ========== String Similarity Features ==========
    
    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Jaccard similarity between two strings (token-based).
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaccard similarity [0, 1]
        """
        if not str1 or not str2:
            return 0.0
        
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def token_overlap(self, str1: str, str2: str) -> Tuple[float, float, int]:
        """
        Token overlap metrics.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Tuple of (overlap_ratio_1, overlap_ratio_2, common_tokens_count)
        """
        if not str1 or not str2:
            return 0.0, 0.0, 0
        
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0, 0.0, 0
        
        common = len(tokens1 & tokens2)
        ratio1 = common / len(tokens1) if tokens1 else 0.0
        ratio2 = common / len(tokens2) if tokens2 else 0.0
        
        return ratio1, ratio2, common
    
    def levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Levenshtein (edit) distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Edit distance
        """
        if not str1:
            return len(str2) if str2 else 0
        if not str2:
            return len(str1)
        
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        
        distances = range(len(str1) + 1)
        
        for i2, char2 in enumerate(str2):
            new_distances = [i2 + 1]
            for i1, char1 in enumerate(str1):
                if char1 == char2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
            distances = new_distances
        
        return distances[-1]
    
    def normalized_levenshtein(self, str1: str, str2: str) -> float:
        """
        Normalized Levenshtein similarity [0, 1].
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Normalized similarity
        """
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        distance = self.levenshtein_distance(str1, str2)
        return 1.0 - (distance / max_len)
    
    def jaro_winkler_similarity(self, str1: str, str2: str) -> float:
        """
        Jaro-Winkler similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaro-Winkler similarity [0, 1]
        """
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        # Jaro similarity
        len1, len2 = len(str1), len(str2)
        max_dist = max(len1, len2) // 2 - 1
        max_dist = max(0, max_dist)
        
        matches = 0
        hash1 = [False] * len1
        hash2 = [False] * len2
        
        for i in range(len1):
            start = max(0, i - max_dist)
            end = min(i + max_dist + 1, len2)
            
            for j in range(start, end):
                if not hash2[j] and str1[i] == str2[j]:
                    hash1[i] = True
                    hash2[j] = True
                    matches += 1
                    break
        
        if matches == 0:
            return 0.0
        
        # Transpositions
        transpositions = 0
        k = 0
        for i in range(len1):
            if hash1[i]:
                while not hash2[k]:
                    k += 1
                if str1[i] != str2[k]:
                    transpositions += 1
                k += 1
        
        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
        
        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if str1[i] == str2[i]:
                prefix += 1
            else:
                break
        
        return jaro + prefix * 0.1 * (1 - jaro)
    
    def character_ngram_similarity(self, str1: str, str2: str, n: int = 3) -> float:
        """
        Character n-gram similarity.
        
        Args:
            str1: First string
            str2: Second string
            n: N-gram size
            
        Returns:
            N-gram similarity [0, 1]
        """
        if not str1 or not str2:
            return 0.0
        
        str1 = str1.lower().replace(" ", "")
        str2 = str2.lower().replace(" ", "")
        
        if len(str1) < n and len(str2) < n:
            return 1.0 if str1 == str2 else 0.0
        
        ngrams1 = [str1[i:i+n] for i in range(max(1, len(str1) - n + 1))]
        ngrams2 = [str2[i:i+n] for i in range(max(1, len(str2) - n + 1))]
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        ngrams1_set = set(ngrams1)
        ngrams2_set = set(ngrams2)
        
        intersection = len(ngrams1_set & ngrams2_set)
        union = len(ngrams1_set | ngrams2_set)
        
        return intersection / union if union > 0 else 0.0
    
    def longest_common_substring_ratio(self, str1: str, str2: str) -> float:
        """
        Ratio of longest common substring to shorter string length.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            LCS ratio [0, 1]
        """
        if not str1 or not str2:
            return 0.0
        
        str1 = str1.lower()
        str2 = str2.lower()
        
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
        
        min_len = min(len(str1), len(str2))
        return max_length / min_len if min_len > 0 else 0.0
    
    # ========== Business Name Features ==========
    
    def extract_legal_suffix(self, name: str) -> str:
        """
        Extract legal suffix from business name.
        
        Args:
            name: Business name
            
        Returns:
            Legal suffix or empty string
        """
        if not name:
            return ""
        
        name_lower = name.lower()
        
        suffixes = [
            'ltd', 'limited', 'corp', 'corporation', 'inc', 'incorporated',
            'pvt', 'private', 'llc', 'llp', 'co', 'company', 'plc',
            'gmbh', 'sa', 'nv', 'bv', 'ag', 'spa', 'srl', 'sarl'
        ]
        
        for suffix in suffixes:
            if re.search(r'\b' + suffix + r'\b', name_lower):
                return suffix
        
        return ""
    
    def name_suffix_match(self, name1: str, name2: str) -> int:
        """
        Check if legal suffixes match.
        
        Args:
            name1: First business name
            name2: Second business name
            
        Returns:
            1 if both have suffix and match, 0 otherwise
        """
        suffix1 = self.extract_legal_suffix(name1)
        suffix2 = self.extract_legal_suffix(name2)
        
        if suffix1 and suffix2 and suffix1 == suffix2:
            return 1
        return 0
    
    def remove_legal_suffix(self, name: str) -> str:
        """
        Remove legal suffix from business name.
        
        Args:
            name: Business name
            
        Returns:
            Name without legal suffix
        """
        if not name:
            return ""
        
        suffixes = [
            'ltd', 'limited', 'corp', 'corporation', 'inc', 'incorporated',
            'pvt', 'private', 'llc', 'llp', 'co', 'company', 'plc',
            'gmbh', 'sa', 'nv', 'bv', 'ag', 'spa', 'srl', 'sarl'
        ]
        
        name_lower = name.lower()
        for suffix in suffixes:
            name_lower = re.sub(r'\b' + suffix + r'\b', '', name_lower)
        
        return ' '.join(name_lower.split())
    
    # ========== Address Features ==========
    
    def extract_numeric_parts(self, address: str) -> List[str]:
        """
        Extract numeric parts from address (house numbers, PIN codes, etc.).
        
        Args:
            address: Address string
            
        Returns:
            List of numeric parts
        """
        if not address:
            return []
        
        # Find all numeric sequences
        numbers = re.findall(r'\d+', address)
        return numbers
    
    def numeric_overlap(self, addr1: str, addr2: str) -> float:
        """
        Overlap of numeric parts in addresses.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            Numeric overlap ratio [0, 1]
        """
        nums1 = set(self.extract_numeric_parts(addr1))
        nums2 = set(self.extract_numeric_parts(addr2))
        
        if not nums1 or not nums2:
            return 0.0
        
        intersection = len(nums1 & nums2)
        union = len(nums1 | nums2)
        
        return intersection / union if union > 0 else 0.0
    
    def address_token_importance(self, address: str) -> Dict[str, float]:
        """
        Assign importance weights to address tokens.
        
        Args:
            address: Address string
            
        Returns:
            Dictionary of token -> importance score
        """
        if not address:
            return {}
        
        # Common unimportant words
        stop_words = {'near', 'opp', 'opposite', 'next', 'beside', 'behind', 'front'}
        
        tokens = address.lower().split()
        weights = {}
        
        for token in tokens:
            if token in stop_words:
                weights[token] = 0.1
            elif token.isdigit():
                weights[token] = 1.0  # Numbers are important
            elif len(token) >= 6:
                weights[token] = 0.8  # Longer words often more specific
            else:
                weights[token] = 0.5
        
        return weights
    
    # ========== Feature Generation ==========
    
    def generate_pair_features(self, 
                              row1: pd.Series, 
                              row2: pd.Series) -> Dict[str, float]:
        """
        Generate all features for a pair of entities.
        
        Args:
            row1: First entity (Source 1)
            row2: Second entity (Source 2 or 3)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Get fields
        name1 = row1.get('business_name_clean', '')
        name2 = row2.get('business_name_clean', '')
        addr1 = row1.get('business_address_clean', '')
        addr2 = row2.get('business_address_clean', '')
        country1 = row1.get('country_clean', '')
        country2 = row2.get('country_clean', '')
        
        # Country match
        features['country_exact_match'] = 1.0 if country1 == country2 else 0.0
        
        # ===== Business Name Features =====
        features['name_jaccard'] = self.jaccard_similarity(name1, name2)
        features['name_levenshtein'] = self.normalized_levenshtein(name1, name2)
        features['name_jaro_winkler'] = self.jaro_winkler_similarity(name1, name2)
        features['name_trigram'] = self.character_ngram_similarity(name1, name2, 3)
        features['name_bigram'] = self.character_ngram_similarity(name1, name2, 2)
        features['name_lcs_ratio'] = self.longest_common_substring_ratio(name1, name2)
        
        # Token overlap
        name_overlap1, name_overlap2, name_common = self.token_overlap(name1, name2)
        features['name_token_overlap_1'] = name_overlap1
        features['name_token_overlap_2'] = name_overlap2
        features['name_common_tokens'] = float(name_common)
        
        # Legal suffix
        features['name_suffix_match'] = float(self.name_suffix_match(name1, name2))
        
        # Name without suffix
        name1_no_suffix = self.remove_legal_suffix(name1)
        name2_no_suffix = self.remove_legal_suffix(name2)
        features['name_no_suffix_jaccard'] = self.jaccard_similarity(name1_no_suffix, name2_no_suffix)
        features['name_no_suffix_levenshtein'] = self.normalized_levenshtein(name1_no_suffix, name2_no_suffix)
        
        # Length features
        features['name_length_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
        features['name_length_diff'] = abs(len(name1) - len(name2))
        
        # ===== Address Features =====
        features['addr_jaccard'] = self.jaccard_similarity(addr1, addr2)
        features['addr_levenshtein'] = self.normalized_levenshtein(addr1, addr2)
        features['addr_jaro_winkler'] = self.jaro_winkler_similarity(addr1, addr2)
        features['addr_trigram'] = self.character_ngram_similarity(addr1, addr2, 3)
        features['addr_lcs_ratio'] = self.longest_common_substring_ratio(addr1, addr2)
        
        # Token overlap
        addr_overlap1, addr_overlap2, addr_common = self.token_overlap(addr1, addr2)
        features['addr_token_overlap_1'] = addr_overlap1
        features['addr_token_overlap_2'] = addr_overlap2
        features['addr_common_tokens'] = float(addr_common)
        
        # Numeric overlap
        features['addr_numeric_overlap'] = self.numeric_overlap(addr1, addr2)
        
        # Length features
        features['addr_length_ratio'] = min(len(addr1), len(addr2)) / max(len(addr1), len(addr2), 1)
        features['addr_length_diff'] = abs(len(addr1) - len(addr2))
        
        # ===== Combined Features =====
        features['avg_name_addr_similarity'] = (
            features['name_jaccard'] + features['addr_jaccard']
        ) / 2.0
        
        features['max_name_similarity'] = max(
            features['name_jaccard'],
            features['name_levenshtein'],
            features['name_jaro_winkler']
        )
        
        features['max_addr_similarity'] = max(
            features['addr_jaccard'],
            features['addr_levenshtein'],
            features['addr_jaro_winkler']
        )
        
        return features
    
    def generate_features_for_pairs(self,
                                   pairs: pd.DataFrame,
                                   source1: pd.DataFrame,
                                   combined: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for a dataframe of pairs.
        
        Args:
            pairs: Dataframe with s1_id and s2s3_id columns
            source1: Source 1 dataframe
            combined: Combined source2+source3 dataframe
            
        Returns:
            Dataframe with features
        """
        # Create lookup dictionaries
        s1_dict = source1.set_index('entity_id').to_dict('index')
        combined_dict = combined.set_index('entity_id').to_dict('index')
        
        logger.info(f"Generating features for {len(pairs)} pairs...")
        
        features_list = []
        
        for idx, row in pairs.iterrows():
            s1_id = row['s1_id']
            s2s3_id = row['s2s3_id']
            
            if s1_id not in s1_dict or s2s3_id not in combined_dict:
                continue
            
            s1_row = pd.Series(s1_dict[s1_id])
            s2s3_row = pd.Series(combined_dict[s2s3_id])
            
            features = self.generate_pair_features(s1_row, s2s3_row)
            features['s1_id'] = s1_id
            features['s2s3_id'] = s2s3_id
            
            if 'label' in row:
                features['label'] = row['label']
            
            features_list.append(features)
            
            if (idx + 1) % 10000 == 0:
                logger.info(f"  Processed {idx + 1} / {len(pairs)} pairs")
        
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"Generated {len(features_df)} feature vectors with {len(features_df.columns)-2} features")
        
        return features_df


def main():
    """Test feature generation."""
    from data_loader import DataLoader
    
    loader = DataLoader()
    
    # Load and preprocess data
    s1_train, s2_train, s3_train = loader.load_source_data("train")
    s1_train = loader.preprocess_dataframe(s1_train)
    s2_train = loader.preprocess_dataframe(s2_train)
    s3_train = loader.preprocess_dataframe(s3_train)
    
    # Create sample pairs
    combined = pd.concat([s2_train, s3_train], ignore_index=True)
    sample_pairs = pd.DataFrame({
        's1_id': [s1_train.iloc[0]['entity_id']] * 3,
        's2s3_id': combined.iloc[:3]['entity_id'].values,
        'label': [1, 0, 0]
    })
    
    # Generate features
    engineer = FeatureEngineer()
    features_df = engineer.generate_features_for_pairs(sample_pairs, s1_train, combined)
    
    print("\nSample features:")
    print(features_df.head())
    print(f"\nFeature columns: {len(features_df.columns)}")


if __name__ == "__main__":
    main()
