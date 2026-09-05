"""
MediVision AI — Full Metrics Evaluation Script
Calculates: per-class AUROC, mean AUROC, inference latency, dataset stats
Requires: pretrained_model.h5 + NIH CSV files in same dir or update paths below
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG — update these paths if needed ──────────────────────────────────────
MODEL_WEIGHTS   = r"C:\Users\mansi\OneDrive\Desktop\Documents\Final Year\medaidmodel\pretrained_model.h5"
TEST_CSV        = r"C:\Users\mansi\OneDrive\Desktop\Documents\Final Year\medaidmodel\Chest-X-Ray-Medical-Diagnosis-master\nih\test.csv"
IMAGE_DIR       = r"C:\Users\mansi\OneDrive\Desktop\Documents\Final Year\medaidmodel\Chest-X-Ray-Medical-Diagnosis-master\nih\images-small"
BATCH_SIZE      = 32
IMAGE_SIZE      = (224, 224)
LATENCY_SAMPLES = 50                  # number of images to time for latency
OUTPUT_CSV      = "metrics_output.csv"
# ──────────────────────────────────────────────────────────────────────────────

LABELS = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]


# ── 1. Build model ─────────────────────────────────────────────────────────────
def build_model(weights_path):
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.densenet import preprocess_input

    print("Building DenseNet-121 model...")
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(len(LABELS), activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(weights_path)
    print(f"  Loaded weights from: {weights_path}")
    return model, preprocess_input


# ── 2. Build test generator ────────────────────────────────────────────────────
def build_test_generator(test_csv, image_dir, preprocess_fn):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    test_df = pd.read_csv(test_csv)
    print(f"\nTest set: {len(test_df)} images")

    gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_generator = gen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col='Image',
        y_col=LABELS,
        class_mode='raw',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return test_generator, test_df


# ── 3. Compute AUROC ───────────────────────────────────────────────────────────
def compute_auroc(model, test_generator):
    from sklearn.metrics import roc_auc_score

    print("\nRunning inference on test set (this may take a few minutes)...")
    predicted = model.predict(test_generator, steps=len(test_generator), verbose=1)
    true_labels = test_generator.labels

    # clip to valid length in case of rounding
    n = min(len(predicted), len(true_labels))
    predicted = predicted[:n]
    true_labels = true_labels[:n]

    results = {}
    for i, label in enumerate(LABELS):
        y_true = true_labels[:, i]
        y_pred = predicted[:, i]
        # skip if only one class present (can't compute AUC)
        if len(np.unique(y_true)) < 2:
            print(f"  Skipping {label} — only one class in test set")
            results[label] = float('nan')
        else:
            auc = roc_auc_score(y_true, y_pred)
            results[label] = round(auc, 4)

    valid_aucs = [v for v in results.values() if not np.isnan(v)]
    mean_auc = round(np.mean(valid_aucs), 4) if valid_aucs else float('nan')
    return results, mean_auc, predicted, true_labels


# ── 4. Inference latency ───────────────────────────────────────────────────────
def compute_latency(model, image_dir, test_df, preprocess_fn, n=LATENCY_SAMPLES):
    from tensorflow.keras.preprocessing import image as keras_image

    print(f"\nMeasuring inference latency over {n} single images...")
    sample_files = test_df['Image'].values[:n]
    times = []

    for fname in sample_files:
        fpath = os.path.join(image_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = keras_image.load_img(fpath, target_size=IMAGE_SIZE)
        arr = keras_image.img_to_array(img)
        arr = preprocess_fn(np.expand_dims(arr, axis=0))

        start = time.perf_counter()
        _ = model.predict(arr, verbose=0)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # convert to ms

    if not times:
        print("  No images found for latency test — check IMAGE_DIR path")
        return None, None

    mean_ms = round(np.mean(times), 2)
    p95_ms  = round(np.percentile(times, 95), 2)
    return mean_ms, p95_ms


# ── 5. Dataset stats ───────────────────────────────────────────────────────────
def compute_dataset_stats(test_csv):
    test_df = pd.read_csv(test_csv)
    stats = {}
    for label in LABELS:
        if label in test_df.columns:
            stats[label] = int(test_df[label].sum())
    total = len(test_df)
    return total, stats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # check files exist
    for path, name in [(MODEL_WEIGHTS, "Model weights"), (TEST_CSV, "Test CSV")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at '{path}'. Update the path in CONFIG.")
            return

    model, preprocess_fn = build_model(MODEL_WEIGHTS)
    test_generator, test_df = build_test_generator(TEST_CSV, IMAGE_DIR, preprocess_fn)

    # AUROC
    auroc_per_class, mean_auroc, predicted, true_labels = compute_auroc(model, test_generator)

    # Latency
    mean_latency_ms, p95_latency_ms = compute_latency(model, IMAGE_DIR, test_df, preprocess_fn)

    # Dataset stats
    total_test, class_counts = compute_dataset_stats(TEST_CSV)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  MediVision AI — Evaluation Results")
    print("="*55)

    print(f"\nDataset: NIH ChestX-ray14 (small subset)")
    print(f"Test images: {total_test}")

    print(f"\n{'Condition':<22} {'AUROC':>8}  {'Test Positives':>15}")
    print("-"*48)
    for label in LABELS:
        auc = auroc_per_class.get(label, float('nan'))
        count = class_counts.get(label, 0)
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A"
        print(f"{label:<22} {auc_str:>8}  {count:>15}")

    print("-"*48)
    print(f"{'Mean AUROC':<22} {mean_auroc:>8.4f}")

    if mean_latency_ms:
        print(f"\nInference latency (single image, CPU/GPU):")
        print(f"  Mean : {mean_latency_ms} ms")
        print(f"  P95  : {p95_latency_ms} ms")

    # ── Save to CSV ────────────────────────────────────────────────────────────
    rows = []
    for label in LABELS:
        rows.append({
            "Condition": label,
            "AUROC": auroc_per_class.get(label, float('nan')),
            "Test_Positives": class_counts.get(label, 0)
        })
    rows.append({"Condition": "MEAN", "AUROC": mean_auroc, "Test_Positives": ""})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print("="*55)


if __name__ == "__main__":
    main()
