import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def augment_dataset(input_file, output_file, target_total_rows=2100):
    print(f"Loading base logic from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Define columns based on your dataset structure
    numeric_cols = ['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    categorical_cols = ['Soil Type', 'Crop Type']
    target_col = 'Fertilizer Name'
    
    # Calculate how many rows we need per class for a perfectly balanced dataset
    classes = df[target_col].unique()
    rows_per_class = target_total_rows // len(classes)
    
    synthetic_data = []
    
    for fert_class in classes:
        # Isolate the logic for this specific fertilizer
        class_df = df[df[target_col] == fert_class]
        
        # 1. Generate Numerical Data (NPK & Environment)
        class_means = class_df[numeric_cols].mean()
        # If a class has zero variance (e.g., Potassium is always 0 for Urea), 
        # we add a tiny bit of noise (std=1.0) so the model learns generalization, not memorization.
        class_stds = class_df[numeric_cols].std().replace(0, 1.0).fillna(1.0) 
        
        # Sample from a normal distribution
        synthetic_num = np.random.normal(
            loc=class_means.values,
            scale=class_stds.values,
            size=(rows_per_class, len(numeric_cols))
        )
        
        # Clip values to make physical sense (e.g., no negative Nitrogen) and round to integers
        synthetic_num = np.clip(synthetic_num, a_min=0, a_max=None)
        synthetic_num = np.round(synthetic_num).astype(int)
        
        # 2. Generate Categorical Data (Soil & Crop)
        synthetic_cat = {}
        for cat_col in categorical_cols:
            # Preserve the probability distribution of crops/soils for this fertilizer
            value_counts = class_df[cat_col].value_counts(normalize=True)
            synthetic_cat[cat_col] = np.random.choice(
                value_counts.index, 
                size=rows_per_class, 
                p=value_counts.values
            )
            
        # 3. Assemble the synthetic rows for this class
        synth_df = pd.DataFrame(synthetic_num, columns=numeric_cols)
        for cat_col in categorical_cols:
            synth_df[cat_col] = synthetic_cat[cat_col]
        synth_df[target_col] = fert_class
        
        synthetic_data.append(synth_df)
        
    # Combine all classes, shuffle the rows randomly, and save
    augmented_df = pd.concat(synthetic_data, ignore_index=True)
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    augmented_df.to_csv(output_file, index=False)
    print(f"Success! Generated a perfectly balanced dataset of {len(augmented_df)} rows.")
    print(f"Saved to: {output_file}")
    
    return augmented_df

# Execute the pipeline
if __name__ == "__main__":
    # Ensure the original file is in the same directory as the script
    augmented_dataset = augment_dataset(
        input_file="D:\laptop\College\GP\Fertilizer_Recommendation\src\Fertilizer dataset.csv", 
        output_file="Fertilizer_Augmented_Data.csv", 
        target_total_rows=2100 # 300 rows per each of the 7 classes
    )