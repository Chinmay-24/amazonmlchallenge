# Quick Setup Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install pandas numpy lightgbm scikit-learn
```

Or use the requirements file:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2: Add Your Dataset

Place the challenge data files in the `dataset/` directory:

```
dataset/
├── train/
│   ├── train_source1.tsv       ← Place here
│   ├── train_source2.tsv       ← Place here
│   ├── train_source3.tsv       ← Place here
│   └── train_ground_truth.tsv  ← Place here
└── test/
    ├── test_source1.tsv        ← Place here
    ├── test_source2.tsv        ← Place here
    └── test_source3.tsv        ← Place here
```

### Step 3: Run the Pipeline

```bash
cd code/business_entity_resolution
python src/main.py --mode full
```

Wait 10-20 minutes for training and inference to complete.

## ✅ Verify Your Submission

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Look for "VALIDATION PASSED ✓" message.

## 📤 Submit

Upload `output/matching_results.tsv` to the challenge portal!

## 🔧 Troubleshooting

**Error: "ModuleNotFoundError: No module named 'lightgbm'"**
→ Run: `pip install lightgbm`

**Error: "FileNotFoundError: dataset/train/train_source1.tsv"**
→ Make sure your data files are in the correct directories

**Validation fails**
→ Check that your files are tab-separated (not comma-separated)

## 📚 Documentation

- **Quick Start**: `README.md`
- **Detailed Guide**: `code/business_entity_resolution/README.md`
- **Methodology Template**: `Documentation_template.md`

## 💡 Tips

1. **First run**: Use a small sample to test the pipeline
2. **Tune threshold**: Adjust in `model.py` for better F_0.5
3. **Add features**: Edit `features.py` to add custom features
4. **Modify blocking**: Edit `blocking.py` to tune candidate generation

## 📊 Expected Output

After running, you should see:

```
output/
├── candidate_pairs.tsv      # ~50-100 candidates per S1 entity
└── matching_results.tsv     # ~1-5 matches per S1 entity

models/
└── entity_matching_model.pkl  # Trained model (~50MB)
```

## ⏱️ Expected Runtime

- Training: 5-15 minutes
- Inference: 2-5 minutes
- Total: ~10-20 minutes

## 🎯 Goals

- **Blocking**: Aim for <100 avg candidates per S1 entity
- **F_0.5 Score**: Aim for >0.85 (precision-weighted metric)
- **Coverage**: 100% of S1 entities in submission

---

**Need help?** Check the detailed README files or review the code comments!
