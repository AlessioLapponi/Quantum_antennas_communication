from pathlib import Path
import joblib

original_path = Path("models/pca_ml_curve_model.joblib")
compressed_path = Path("models/pca_ml_curve_model_compressed.joblib")

print("Loading original model...")
model = joblib.load(original_path)

print("Saving compressed model...")
joblib.dump(model, compressed_path, compress=3)

original_size = original_path.stat().st_size / (1024 * 1024)
compressed_size = compressed_path.stat().st_size / (1024 * 1024)

print(f"Original size:   {original_size:.2f} MB")
print(f"Compressed size: {compressed_size:.2f} MB")