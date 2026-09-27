"""
Validation script for submission files.
Checks formatting, completeness, and consistency of both output files.
Based on the official validation requirements from the challenge.
"""

import sys
import argparse
from pathlib import Path
from typing import Set, List, Tuple


def read_tsv_file(filepath: Path) -> List[Tuple[str, str]]:
    """
    Read TSV file and return list of (col1, col2) tuples.
    
    Args:
        filepath: Path to TSV file
        
    Returns:
        List of tuples
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        if not lines:
            raise ValueError(f"File {filepath} is empty")
        
        # Check header
        header = lines[0].strip().split('\t')
        
        # Read data rows
        for i, line in enumerate(lines[1:], start=2):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                raise ValueError(f"Line {i}: Expected 2 columns, got {len(parts)}")
            rows.append((parts[0], parts[1]))
    
    return rows


def load_source_ids(test_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Load valid entity IDs from test source files.
    
    Args:
        test_dir: Path to test directory
        
    Returns:
        Tuple of (source1_ids, source2_ids, source3_ids)
    """
    source1_ids = set()
    source2_ids = set()
    source3_ids = set()
    
    # Load Source 1
    s1_file = test_dir / "test_source1.tsv"
    if not s1_file.exists():
        raise FileNotFoundError(f"Test file not found: {s1_file}")
    
    with open(s1_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                source1_ids.add(parts[0])
    
    # Load Source 2
    s2_file = test_dir / "test_source2.tsv"
    if s2_file.exists():
        with open(s2_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    source2_ids.add(parts[0])
    
    # Load Source 3
    s3_file = test_dir / "test_source3.tsv"
    if s3_file.exists():
        with open(s3_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    source3_ids.add(parts[0])
    
    return source1_ids, source2_ids, source3_ids


def parse_id_list(id_str: str) -> List[str]:
    """
    Parse comma-separated ID list.
    
    Args:
        id_str: Comma-separated IDs
        
    Returns:
        List of IDs
    """
    if not id_str or not id_str.strip():
        return []
    
    return [id.strip() for id in id_str.split(',')]


def validate_file(filepath: Path, 
                 expected_col1: str,
                 expected_col2: str) -> Tuple[bool, List[str], List[Tuple[str, str]]]:
    """
    Validate basic file format.
    
    Args:
        filepath: Path to file
        expected_col1: Expected first column name
        expected_col2: Expected second column name
        
    Returns:
        Tuple of (is_valid, errors, data_rows)
    """
    errors = []
    
    # Check file exists
    if not filepath.exists():
        errors.append(f"File not found: {filepath}")
        return False, errors, []
    
    # Check file is not empty
    if filepath.stat().st_size == 0:
        errors.append(f"File is empty: {filepath}")
        return False, errors, []
    
    # Read and validate
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            if not lines:
                errors.append("File contains no lines")
                return False, errors, []
            
            # Check header
            header = lines[0].strip().split('\t')
            if len(header) != 2:
                errors.append(f"Header must have exactly 2 columns, got {len(header)}")
                return False, errors, []
            
            if header[0] != expected_col1:
                errors.append(f"First column should be '{expected_col1}', got '{header[0]}'")
            
            if header[1] != expected_col2:
                errors.append(f"Second column should be '{expected_col2}', got '{header[1]}'")
            
            # Read data rows
            data_rows = []
            for i, line in enumerate(lines[1:], start=2):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    errors.append(f"Line {i}: Expected 2 columns, got {len(parts)}")
                    continue
                data_rows.append((parts[0], parts[1]))
            
            if not data_rows:
                errors.append("File contains no data rows (only header)")
                return False, errors, []
    
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors, []
    
    return len(errors) == 0, errors, data_rows


def validate_submission(matching_file: Path,
                       candidate_file: Path,
                       test_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate both submission files.
    
    Args:
        matching_file: Path to matching_results.tsv
        candidate_file: Path to candidate_pairs.tsv
        test_dir: Path to test data directory
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    all_errors = []
    
    print("=" * 70)
    print("VALIDATING SUBMISSION FILES")
    print("=" * 70)
    
    # Validate matching_results.tsv format
    print("\n[1] Validating matching_results.tsv format...")
    is_valid, errors, matching_rows = validate_file(
        matching_file, 
        'source1_entity_id',
        'matched_entity_ids'
    )
    
    if errors:
        all_errors.extend([f"matching_results.tsv: {e}" for e in errors])
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print(f"  ✓ Format valid ({len(matching_rows)} rows)")
    
    # Validate candidate_pairs.tsv format
    print("\n[2] Validating candidate_pairs.tsv format...")
    is_valid, errors, candidate_rows = validate_file(
        candidate_file,
        'source1_entity_id',
        'candidate_entity_ids'
    )
    
    if errors:
        all_errors.extend([f"candidate_pairs.tsv: {e}" for e in errors])
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print(f"  ✓ Format valid ({len(candidate_rows)} rows)")
    
    # If format errors, stop here
    if all_errors:
        return False, all_errors
    
    # Load valid IDs from test files
    print("\n[3] Loading test data...")
    try:
        source1_ids, source2_ids, source3_ids = load_source_ids(test_dir)
        valid_target_ids = source2_ids | source3_ids
        print(f"  ✓ Found {len(source1_ids)} S1, {len(source2_ids)} S2, {len(source3_ids)} S3 entities")
    except Exception as e:
        all_errors.append(f"Error loading test data: {e}")
        return False, all_errors
    
    # Parse data into dictionaries
    matching_dict = {}
    candidate_dict = {}
    
    for s1_id, matched_ids_str in matching_rows:
        matching_dict[s1_id] = parse_id_list(matched_ids_str)
    
    for s1_id, candidate_ids_str in candidate_rows:
        candidate_dict[s1_id] = parse_id_list(candidate_ids_str)
    
    # Validation checks
    print("\n[4] Checking completeness...")
    
    # Check 4a: All S1 entities present in both files
    matching_s1_ids = set(matching_dict.keys())
    candidate_s1_ids = set(candidate_dict.keys())
    
    if matching_s1_ids != source1_ids:
        missing = source1_ids - matching_s1_ids
        extra = matching_s1_ids - source1_ids
        if missing:
            all_errors.append(f"matching_results.tsv: Missing {len(missing)} S1 entities")
            print(f"  ✗ Missing S1 entities in matching_results.tsv: {len(missing)}")
        if extra:
            all_errors.append(f"matching_results.tsv: {len(extra)} unknown S1 entities")
            print(f"  ✗ Unknown S1 entities in matching_results.tsv: {len(extra)}")
    else:
        print(f"  ✓ All {len(source1_ids)} S1 entities present in matching_results.tsv")
    
    if candidate_s1_ids != source1_ids:
        missing = source1_ids - candidate_s1_ids
        extra = candidate_s1_ids - source1_ids
        if missing:
            all_errors.append(f"candidate_pairs.tsv: Missing {len(missing)} S1 entities")
            print(f"  ✗ Missing S1 entities in candidate_pairs.tsv: {len(missing)}")
        if extra:
            all_errors.append(f"candidate_pairs.tsv: {len(extra)} unknown S1 entities")
            print(f"  ✗ Unknown S1 entities in candidate_pairs.tsv: {len(extra)}")
    else:
        print(f"  ✓ All {len(source1_ids)} S1 entities present in candidate_pairs.tsv")
    
    # Check 4b: No duplicate rows
    if len(matching_rows) != len(matching_dict):
        dup_count = len(matching_rows) - len(matching_dict)
        all_errors.append(f"matching_results.tsv: {dup_count} duplicate S1 entity rows")
        print(f"  ✗ Duplicate rows in matching_results.tsv: {dup_count}")
    
    if len(candidate_rows) != len(candidate_dict):
        dup_count = len(candidate_rows) - len(candidate_dict)
        all_errors.append(f"candidate_pairs.tsv: {dup_count} duplicate S1 entity rows")
        print(f"  ✗ Duplicate rows in candidate_pairs.tsv: {dup_count}")
    
    print("\n[5] Checking ID validity...")
    
    # Check 5a: All target IDs are valid S2/S3 IDs
    invalid_matching = 0
    invalid_candidate = 0
    
    for s1_id, target_ids in matching_dict.items():
        for target_id in target_ids:
            if target_id not in valid_target_ids:
                invalid_matching += 1
                all_errors.append(f"matching_results.tsv: Invalid target ID '{target_id}' for {s1_id}")
                if invalid_matching >= 10:  # Limit error messages
                    break
        if invalid_matching >= 10:
            break
    
    for s1_id, target_ids in candidate_dict.items():
        for target_id in target_ids:
            if target_id not in valid_target_ids:
                invalid_candidate += 1
                all_errors.append(f"candidate_pairs.tsv: Invalid target ID '{target_id}' for {s1_id}")
                if invalid_candidate >= 10:  # Limit error messages
                    break
        if invalid_candidate >= 10:
            break
    
    if invalid_matching > 0:
        print(f"  ✗ Invalid target IDs in matching_results.tsv: {invalid_matching}+")
    else:
        print(f"  ✓ All target IDs valid in matching_results.tsv")
    
    if invalid_candidate > 0:
        print(f"  ✗ Invalid target IDs in candidate_pairs.tsv: {invalid_candidate}+")
    else:
        print(f"  ✓ All target IDs valid in candidate_pairs.tsv")
    
    print("\n[6] Checking for duplicates within ID lists...")
    
    # Check 6: No duplicate IDs within lists
    dup_matching = 0
    dup_candidate = 0
    
    for s1_id, target_ids in matching_dict.items():
        if len(target_ids) != len(set(target_ids)):
            dup_matching += 1
            if dup_matching <= 5:
                all_errors.append(f"matching_results.tsv: Duplicate IDs in list for {s1_id}")
    
    for s1_id, target_ids in candidate_dict.items():
        if len(target_ids) != len(set(target_ids)):
            dup_candidate += 1
            if dup_candidate <= 5:
                all_errors.append(f"candidate_pairs.tsv: Duplicate IDs in list for {s1_id}")
    
    if dup_matching > 0:
        print(f"  ✗ Entities with duplicate IDs in matching_results.tsv: {dup_matching}")
    else:
        print(f"  ✓ No duplicate IDs within lists in matching_results.tsv")
    
    if dup_candidate > 0:
        print(f"  ✗ Entities with duplicate IDs in candidate_pairs.tsv: {dup_candidate}")
    else:
        print(f"  ✓ No duplicate IDs within lists in candidate_pairs.tsv")
    
    print("\n[7] Checking consistency between files...")
    
    # Check 7: All matched IDs should be in candidate IDs
    inconsistent = 0
    for s1_id in matching_dict.keys():
        matched_set = set(matching_dict.get(s1_id, []))
        candidate_set = set(candidate_dict.get(s1_id, []))
        
        not_in_candidates = matched_set - candidate_set
        if not_in_candidates:
            inconsistent += 1
            if inconsistent <= 5:
                all_errors.append(
                    f"Inconsistency: {s1_id} has {len(not_in_candidates)} "
                    f"matched IDs not in candidates"
                )
    
    if inconsistent > 0:
        print(f"  ✗ Entities with matched IDs not in candidates: {inconsistent}")
    else:
        print(f"  ✓ All matched IDs are present in candidates")
    
    # Summary
    print("\n" + "=" * 70)
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} error(s) found")
        print("=" * 70)
        print("\nErrors:")
        for i, error in enumerate(all_errors[:20], 1):
            print(f"  {i}. {error}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors")
        return False, all_errors
    else:
        print("VALIDATION PASSED ✓")
        print("=" * 70)
        
        # Statistics
        total_matches = sum(len(v) for v in matching_dict.values())
        total_candidates = sum(len(v) for v in candidate_dict.values())
        avg_matches = total_matches / len(matching_dict) if matching_dict else 0
        avg_candidates = total_candidates / len(candidate_dict) if candidate_dict else 0
        
        print(f"\nStatistics:")
        print(f"  S1 entities: {len(source1_ids)}")
        print(f"  Entities with matches: {sum(1 for v in matching_dict.values() if v)}")
        print(f"  Singletons: {sum(1 for v in matching_dict.values() if not v)}")
        print(f"  Total matches: {total_matches} (avg: {avg_matches:.2f} per S1)")
        print(f"  Total candidates: {total_candidates} (avg: {avg_candidates:.1f} per S1)")
        
        return True, []


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission files for Amazon ML Challenge"
    )
    parser.add_argument(
        "--matching",
        required=True,
        help="Path to matching_results.tsv"
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate_pairs.tsv"
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to test data directory"
    )
    
    args = parser.parse_args()
    
    matching_file = Path(args.matching)
    candidate_file = Path(args.candidate)
    test_dir = Path(args.test_dir)
    
    is_valid, errors = validate_submission(matching_file, candidate_file, test_dir)
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
