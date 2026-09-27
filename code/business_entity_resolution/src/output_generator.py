"""
Generate submission outputs in TSV format.
Creates both matching_results.tsv and candidate_pairs.tsv.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generate submission outputs in correct TSV format."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize output generator.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_id_list(self, id_list: List[str]) -> str:
        """
        Format list of IDs as comma-separated string.
        
        Args:
            id_list: List of entity IDs
            
        Returns:
            Comma-separated string (empty string for empty list)
        """
        if not id_list:
            return ""
        
        # Remove duplicates and sort
        unique_ids = sorted(set(id_list))
        return ",".join(unique_ids)
    
    def candidates_to_dataframe(self, candidates: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Convert candidate dictionary to dataframe.
        
        Args:
            candidates: Dictionary mapping s1_id -> list of candidate ids
            
        Returns:
            Dataframe with source1_entity_id and candidate_entity_ids columns
        """
        rows = []
        
        for s1_id in sorted(candidates.keys()):
            cand_list = candidates[s1_id]
            rows.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': self.format_id_list(cand_list)
            })
        
        return pd.DataFrame(rows)
    
    def matches_to_dataframe(self, matches: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Convert matches dictionary to dataframe.
        
        Args:
            matches: Dictionary mapping s1_id -> list of matched ids
            
        Returns:
            Dataframe with source1_entity_id and matched_entity_ids columns
        """
        rows = []
        
        for s1_id in sorted(matches.keys()):
            match_list = matches[s1_id]
            rows.append({
                'source1_entity_id': s1_id,
                'matched_entity_ids': self.format_id_list(match_list)
            })
        
        return pd.DataFrame(rows)
    
    def save_candidate_pairs(self, 
                            candidates: Dict[str, List[str]],
                            filename: str = "candidate_pairs.tsv"):
        """
        Save candidate pairs to TSV file.
        
        Args:
            candidates: Dictionary mapping s1_id -> list of candidate ids
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        df = self.candidates_to_dataframe(candidates)
        
        # Save with tab separator, no index, no quoting
        df.to_csv(output_path, sep="\t", index=False, quoting=3)  # quoting=3 is QUOTE_NONE
        
        logger.info(f"Saved candidate pairs to {output_path}")
        logger.info(f"  Total S1 entities: {len(df)}")
        logger.info(f"  Entities with candidates: {sum(df['candidate_entity_ids'] != '')}")
        
        total_candidates = sum(len(v) for v in candidates.values())
        avg_candidates = total_candidates / len(candidates) if candidates else 0
        logger.info(f"  Total candidate pairs: {total_candidates}")
        logger.info(f"  Average candidates per S1: {avg_candidates:.1f}")
    
    def save_matching_results(self,
                             matches: Dict[str, List[str]],
                             filename: str = "matching_results.tsv"):
        """
        Save matching results to TSV file.
        
        Args:
            matches: Dictionary mapping s1_id -> list of matched ids
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        df = self.matches_to_dataframe(matches)
        
        # Save with tab separator, no index, no quoting
        df.to_csv(output_path, sep="\t", index=False, quoting=3)  # quoting=3 is QUOTE_NONE
        
        logger.info(f"Saved matching results to {output_path}")
        logger.info(f"  Total S1 entities: {len(df)}")
        logger.info(f"  Entities with matches: {sum(df['matched_entity_ids'] != '')}")
        logger.info(f"  Singletons: {sum(df['matched_entity_ids'] == '')}")
        
        total_matches = sum(len(v) for v in matches.values())
        avg_matches = total_matches / len(matches) if matches else 0
        logger.info(f"  Total matches: {total_matches}")
        logger.info(f"  Average matches per S1: {avg_matches:.2f}")
    
    def verify_submission_format(self,
                                candidates: Dict[str, List[str]],
                                matches: Dict[str, List[str]],
                                source1_ids: List[str],
                                source2_ids: List[str],
                                source3_ids: List[str]) -> List[str]:
        """
        Verify submission format and consistency.
        
        Args:
            candidates: Candidate pairs dictionary
            matches: Matching results dictionary
            source1_ids: List of valid Source 1 IDs
            source2_ids: List of valid Source 2 IDs
            source3_ids: List of valid Source 3 IDs
            
        Returns:
            List of error messages (empty if no errors)
        """
        errors = []
        
        source1_set = set(source1_ids)
        valid_target_ids = set(source2_ids) | set(source3_ids)
        
        # Check 1: All S1 entities must be present
        candidate_s1_ids = set(candidates.keys())
        matches_s1_ids = set(matches.keys())
        
        if candidate_s1_ids != source1_set:
            missing = source1_set - candidate_s1_ids
            extra = candidate_s1_ids - source1_set
            if missing:
                errors.append(f"Missing S1 entities in candidates: {len(missing)} entities")
            if extra:
                errors.append(f"Extra S1 entities in candidates: {len(extra)} entities")
        
        if matches_s1_ids != source1_set:
            missing = source1_set - matches_s1_ids
            extra = matches_s1_ids - source1_set
            if missing:
                errors.append(f"Missing S1 entities in matches: {len(missing)} entities")
            if extra:
                errors.append(f"Extra S1 entities in matches: {len(extra)} entities")
        
        # Check 2: All matched IDs must be in candidate IDs
        for s1_id in matches.keys():
            matched_set = set(matches.get(s1_id, []))
            candidate_set = set(candidates.get(s1_id, []))
            
            not_in_candidates = matched_set - candidate_set
            if not_in_candidates:
                errors.append(f"Entity {s1_id}: {len(not_in_candidates)} matched IDs not in candidates")
        
        # Check 3: All target IDs must be valid S2/S3 IDs
        for s1_id in candidates.keys():
            for target_id in candidates[s1_id]:
                if target_id not in valid_target_ids:
                    errors.append(f"Invalid target ID in candidates: {target_id}")
                    break  # Only report once per entity
        
        for s1_id in matches.keys():
            for target_id in matches[s1_id]:
                if target_id not in valid_target_ids:
                    errors.append(f"Invalid target ID in matches: {target_id}")
                    break  # Only report once per entity
        
        # Check 4: No duplicate IDs within lists
        for s1_id in candidates.keys():
            cand_list = candidates[s1_id]
            if len(cand_list) != len(set(cand_list)):
                errors.append(f"Duplicate IDs in candidates for {s1_id}")
        
        for s1_id in matches.keys():
            match_list = matches[s1_id]
            if len(match_list) != len(set(match_list)):
                errors.append(f"Duplicate IDs in matches for {s1_id}")
        
        if errors:
            logger.warning(f"Found {len(errors)} verification errors:")
            for error in errors[:10]:  # Show first 10
                logger.warning(f"  - {error}")
            if len(errors) > 10:
                logger.warning(f"  ... and {len(errors) - 10} more")
        else:
            logger.info("✓ All verification checks passed")
        
        return errors
    
    def generate_predictions_from_model(self,
                                       candidates: Dict[str, List[str]],
                                       features_df: pd.DataFrame,
                                       predictions: np.ndarray,
                                       threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Generate match predictions from model output.
        
        Args:
            candidates: Candidate pairs dictionary
            features_df: Features dataframe with s1_id and s2s3_id
            predictions: Model predictions (probabilities or binary)
            threshold: Classification threshold
            
        Returns:
            Dictionary mapping s1_id -> list of matched ids
        """
        # Create predictions dataframe
        pred_df = features_df[['s1_id', 's2s3_id']].copy()
        pred_df['score'] = predictions
        
        # Apply threshold
        if predictions.max() <= 1.0:  # Probabilities
            pred_df['match'] = (pred_df['score'] >= threshold).astype(int)
        else:  # Already binary
            pred_df['match'] = predictions.astype(int)
        
        # Group by s1_id and collect matches
        matches = {}
        
        for s1_id in candidates.keys():
            s1_preds = pred_df[pred_df['s1_id'] == s1_id]
            matched_ids = s1_preds[s1_preds['match'] == 1]['s2s3_id'].tolist()
            matches[s1_id] = matched_ids
        
        return matches


def main():
    """Test output generation."""
    # Create sample data
    candidates = {
        'S1-00001': ['S2-00047', 'S2-00193', 'S3-00812'],
        'S1-00002': ['S3-00004'],
        'S1-00003': [],
        'S1-00004': ['S2-00100', 'S2-00200']
    }
    
    matches = {
        'S1-00001': ['S2-00047', 'S3-00812'],
        'S1-00002': ['S3-00004'],
        'S1-00003': [],
        'S1-00004': ['S2-00100']
    }
    
    # Generate outputs
    generator = OutputGenerator()
    
    generator.save_candidate_pairs(candidates)
    generator.save_matching_results(matches)
    
    # Verify
    source1_ids = ['S1-00001', 'S1-00002', 'S1-00003', 'S1-00004']
    source2_ids = ['S2-00047', 'S2-00193', 'S2-00100', 'S2-00200']
    source3_ids = ['S3-00812', 'S3-00004']
    
    errors = generator.verify_submission_format(
        candidates, matches, source1_ids, source2_ids, source3_ids
    )
    
    print(f"\nVerification: {'PASS' if not errors else 'FAIL'}")


if __name__ == "__main__":
    main()
