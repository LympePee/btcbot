import os
import pandas as pd

# Ο φάκελος που περιέχει τα αρχεία .csv
csv_dir = "data/training_sets"


# Για κάθε αρχείο στο directory
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_dir, filename)
        try:
            df = pd.read_csv(file_path)

            if "timestamp" not in df.columns and "datetime" in df.columns:
                # Προσθέτουμε timestamp σε milliseconds
                df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6
                out_path = file_path.replace(".csv", "_fixed.csv")
                df.to_csv(out_path, index=False)
                print(f"✅ {filename} → {os.path.basename(out_path)} (timestamp added)")
            elif "timestamp" in df.columns:
                print(f"✅ {filename} already contains 'timestamp'")
            else:
                print(f"⚠️ {filename} has no 'datetime' column")
        except Exception as e:
            print(f"❌ Error in {filename}: {e}")
