import os
import glob
import json
import csv

def update_summary():
    results_dir = "results"
    json_files = glob.glob(os.path.join(results_dir, "*_eval.json"))
    
    rows = []
    # Sort files so baseline comes before mctf, and by category
    for jf in sorted(json_files):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            category = data.get("category", "unknown")
            model = "MCTF-ViT" if data.get("model") == "mctf" else "Baseline ViT"
            tokens = int(round(196 * (1.0 - data.get("token_reduction_pct", 0.0) / 100.0)))
            token_red = f"{data.get('token_reduction_pct', 0.0):.2f}%"
            latency = f"{data.get('avg_latency_ms', 0.0):.2f}"
            
            # Compute speedup if baseline exists
            speedup = "1.00x"
            if data.get("model") == "mctf":
                base_jf = os.path.join(results_dir, f"baseline_{category}_eval.json")
                if os.path.exists(base_jf):
                    with open(base_jf, "r") as bf:
                        base_d = json.load(bf)
                        base_lat = base_d.get("avg_latency_ms", 1.0)
                        speedup = f"{(base_lat / max(data.get('avg_latency_ms', 1.0), 0.001)):.2f}x"
            
            img_auroc = f"{(data.get('image_auroc', 0.0) * 100):.2f}%"
            px_auroc = f"{(data.get('pixel_auroc', 0.0) * 100):.2f}%"
            
            rows.append({
                "Category": category,
                "Model": model,
                "Tokens": tokens,
                "Token_Reduction_Pct": token_red,
                "Avg_Latency_ms": latency,
                "Speedup": speedup,
                "Image_AUROC": img_auroc,
                "Pixel_AUROC": px_auroc
            })
        except Exception as e:
            print(f"Error parsing {jf}: {e}")

    # Write CSV
    csv_path = os.path.join(results_dir, "summary.csv")
    fieldnames = ["Category", "Model", "Tokens", "Token_Reduction_Pct", "Avg_Latency_ms", "Speedup", "Image_AUROC", "Pixel_AUROC"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            
    print(f"Updated {csv_path} with {len(rows)} evaluations across categories.")

if __name__ == "__main__":
    update_summary()
