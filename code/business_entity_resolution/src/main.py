"""
Main execution script for entity resolution pipeline.
Orchestrates the entire workflow: blocking -> feature engineering -> training -> inference -> output.
"""

import argparse
import logging
import time
from pathlib import Path
import sys

from data_loader import DataLoader
from blocking import BlockingStrategy
from features import FeatureEngineer
from model import EntityMatchingModel, compute_f_half_macro
from output_generator import OutputGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_pipeline(data_dir: str = "dataset",
                  model_path: str = "models/entity_matching_model.pkl",
                  use_validation_split: bool = True,
                  optimize_threshold: bool = True):
    """
    Train the entity matching model.
    
    Args:
        data_dir: Directory containing training data
        model_path: Path to save trained model
        use_validation_split: Whether to use validation split
        optimize_threshold: Whether to optimize classification threshold
    """
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize components
    loader = DataLoader(data_dir)
    blocker = BlockingStrategy(
        use_country_blocking=True,
        use_token_blocking=True,
        use_ngram_blocking=True,
        use_phonetic_blocking=True
    )
    engineer = FeatureEngineer()
    model = EntityMatchingModel()
    
    # Load training data
    logger.info("\n[1/7] Loading training data...")
    s1_train, s2_train, s3_train = loader.load_source_data("train")
    gt_train = loader.load_ground_truth()
    
    # Preprocess
    logger.info("\n[2/7] Preprocessing data...")
    s1_train = loader.preprocess_dataframe(s1_train)
    s2_train = loader.preprocess_dataframe(s2_train)
    s3_train = loader.preprocess_dataframe(s3_train)
    
    # Parse ground truth
    gt_dict = loader.parse_ground_truth(gt_train)
    
    # Combine sources
    combined_train = loader.combine_sources(s2_train, s3_train)
    
    # Generate candidates (blocking)
    logger.info("\n[3/7] Generating candidate pairs (blocking)...")
    candidates = blocker.generate_candidates(s1_train, s2_train, s3_train)
    
    # Evaluate blocking recall
    total_true_matches = sum(len(v) for v in gt_dict.values())
    total_candidates = sum(len(v) for v in candidates.values())
    
    found_matches = 0
    for s1_id, true_matches in gt_dict.items():
        cand_set = set(candidates.get(s1_id, []))
        found_matches += len([m for m in true_matches if m in cand_set])
    
    blocking_recall = found_matches / total_true_matches if total_true_matches > 0 else 0
    logger.info(f"\nBlocking recall: {blocking_recall:.4f} ({found_matches}/{total_true_matches} true matches found)")
    
    # Create training pairs from candidates
    logger.info("\n[4/7] Creating training pairs...")
    train_pairs = []
    
    for s1_id, cand_list in candidates.items():
        true_matches = set(gt_dict.get(s1_id, []))
        
        for cand_id in cand_list:
            label = 1 if cand_id in true_matches else 0
            train_pairs.append({
                's1_id': s1_id,
                's2s3_id': cand_id,
                'label': label
            })
    
    import pandas as pd
    pairs_df = pd.DataFrame(train_pairs)
    logger.info(f"Created {len(pairs_df)} training pairs: "
               f"{sum(pairs_df['label']==1)} positive, {sum(pairs_df['label']==0)} negative")
    
    # Generate features
    logger.info("\n[5/7] Generating features...")
    features_df = engineer.generate_features_for_pairs(pairs_df, s1_train, combined_train)
    
    # Prepare training data
    X = features_df.drop(['s1_id', 's2s3_id', 'label'], axis=1)
    y = features_df['label']
    
    # Train model
    logger.info("\n[6/7] Training model...")
    
    if use_validation_split:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            num_boost_round=1000,
            early_stopping_rounds=50
        )
        
        # Optimize threshold on validation set
        if optimize_threshold:
            logger.info("\n[7/7] Optimizing classification threshold...")
            model.optimize_threshold(X_val, y_val, metric='f_0.5')
        
        # Final evaluation
        logger.info("\nFinal validation metrics:")
        metrics = model.evaluate(X_val, y_val)
    else:
        history = model.train(
            X, y,
            num_boost_round=500
        )
    
    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    
    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed:.1f} seconds")
    logger.info("=" * 80)


def inference_pipeline(data_dir: str = "dataset",
                      model_path: str = "models/entity_matching_model.pkl",
                      output_dir: str = "output"):
    """
    Run inference on test data and generate submission files.
    
    Args:
        data_dir: Directory containing test data
        model_path: Path to trained model
        output_dir: Directory to save output files
    """
    logger.info("=" * 80)
    logger.info("INFERENCE PIPELINE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize components
    loader = DataLoader(data_dir)
    blocker = BlockingStrategy(
        use_country_blocking=True,
        use_token_blocking=True,
        use_ngram_blocking=True,
        use_phonetic_blocking=True
    )
    engineer = FeatureEngineer()
    model = EntityMatchingModel()
    generator = OutputGenerator(output_dir)
    
    # Load model
    logger.info("\n[1/6] Loading trained model...")
    model.load(model_path)
    
    # Load test data
    logger.info("\n[2/6] Loading test data...")
    s1_test, s2_test, s3_test = loader.load_source_data("test")
    
    # Preprocess
    logger.info("\n[3/6] Preprocessing data...")
    s1_test = loader.preprocess_dataframe(s1_test)
    s2_test = loader.preprocess_dataframe(s2_test)
    s3_test = loader.preprocess_dataframe(s3_test)
    
    # Combine sources
    combined_test = loader.combine_sources(s2_test, s3_test)
    
    # Generate candidates (blocking)
    logger.info("\n[4/6] Generating candidate pairs (blocking)...")
    candidates = blocker.generate_candidates(s1_test, s2_test, s3_test)
    
    # Create inference pairs from candidates
    inference_pairs = []
    for s1_id, cand_list in candidates.items():
        for cand_id in cand_list:
            inference_pairs.append({
                's1_id': s1_id,
                's2s3_id': cand_id
            })
    
    import pandas as pd
    import numpy as np
    
    if inference_pairs:
        pairs_df = pd.DataFrame(inference_pairs)
        logger.info(f"Created {len(pairs_df)} candidate pairs for inference")
        
        # Generate features
        logger.info("\n[5/6] Generating features and running inference...")
        features_df = engineer.generate_features_for_pairs(pairs_df, s1_test, combined_test)
        
        # Run inference
        X = features_df.drop(['s1_id', 's2s3_id'], axis=1)
        predictions = model.predict_proba(X)
        
        # Generate matches from predictions
        matches = generator.generate_predictions_from_model(
            candidates,
            features_df,
            predictions,
            threshold=model.threshold
        )
    else:
        logger.info("No candidate pairs generated")
        matches = {row['entity_id']: [] for _, row in s1_test.iterrows()}
    
    # Ensure all S1 entities are present
    for _, row in s1_test.iterrows():
        s1_id = row['entity_id']
        if s1_id not in candidates:
            candidates[s1_id] = []
        if s1_id not in matches:
            matches[s1_id] = []
    
    # Generate outputs
    logger.info("\n[6/6] Generating output files...")
    generator.save_candidate_pairs(candidates)
    generator.save_matching_results(matches)
    
    # Verify submission
    logger.info("\nVerifying submission format...")
    s1_ids = s1_test['entity_id'].tolist()
    s2_ids = s2_test['entity_id'].tolist()
    s3_ids = s3_test['entity_id'].tolist()
    
    errors = generator.verify_submission_format(
        candidates, matches, s1_ids, s2_ids, s3_ids
    )
    
    if errors:
        logger.error(f"Submission validation failed with {len(errors)} errors!")
        for error in errors[:10]:
            logger.error(f"  - {error}")
    else:
        logger.info("✓ Submission validation passed!")
    
    elapsed = time.time() - start_time
    logger.info(f"\nInference completed in {elapsed:.1f} seconds")
    logger.info("=" * 80)


def full_pipeline(data_dir: str = "dataset",
                 model_path: str = "models/entity_matching_model.pkl",
                 output_dir: str = "output"):
    """
    Run full pipeline: train -> inference -> output.
    
    Args:
        data_dir: Directory containing data
        model_path: Path to save/load model
        output_dir: Directory to save output files
    """
    logger.info("=" * 80)
    logger.info("FULL ENTITY RESOLUTION PIPELINE")
    logger.info("=" * 80)
    
    # Train
    train_pipeline(data_dir, model_path)
    
    # Inference
    inference_pipeline(data_dir, model_path, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nOutput files generated:")
    logger.info(f"  - {output_dir}/candidate_pairs.tsv")
    logger.info(f"  - {output_dir}/matching_results.tsv")
    logger.info(f"\nTo validate submission, run:")
    logger.info(f"  python utils/validate_submission.py \\")
    logger.info(f"    --matching {output_dir}/matching_results.tsv \\")
    logger.info(f"    --candidate {output_dir}/candidate_pairs.tsv \\")
    logger.info(f"    --test-dir {data_dir}/test")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Entity Resolution Pipeline for Amazon ML Challenge"
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'inference', 'full'],
        default='full',
        help='Pipeline mode: train only, inference only, or full pipeline'
    )
    parser.add_argument(
        '--data-dir',
        default='dataset',
        help='Directory containing data (default: dataset)'
    )
    parser.add_argument(
        '--model-path',
        default='models/entity_matching_model.pkl',
        help='Path to save/load model (default: models/entity_matching_model.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Directory to save output files (default: output)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train_pipeline(args.data_dir, args.model_path)
        elif args.mode == 'inference':
            inference_pipeline(args.data_dir, args.model_path, args.output_dir)
        else:  # full
            full_pipeline(args.data_dir, args.model_path, args.output_dir)
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
