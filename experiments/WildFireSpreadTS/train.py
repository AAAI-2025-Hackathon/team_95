import rasterio
import numpy as np
import os
import cv2  
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_absolute_error # For MAE


class WildfireXGBoostDataset:
    def __init__(self, images_dict, sequence_length=7, bands=23, dataset_name=""): 
        self.dates = sorted(images_dict.keys())
        self.images = images_dict
        self.sequence_length = sequence_length
        self.input_features = []
        self.target_values = []  # Now will store DIRECT active fire band values (float)
        self.bands = bands
        self.image_height = None
        self.image_width = None
        self.dataset_name = dataset_name # Store dataset name

        if len(self.dates) < sequence_length + 1:
            raise ValueError("Not enough images to create sequences with specified length.")

        valid_sequences_exist = True

        # --- For inspection: Collect unique values from active fire band ---
        unique_fire_band_values = set()

        nan_target_count = 0 # Counter for NaN targets
        non_nan_target_count = 0 # Counter for non-NaN targets

        for i in range(len(self.dates) - sequence_length):
            seq_dates = self.dates[i:i + sequence_length]
            target_date = self.dates[i + sequence_length]

            sequence = [self.images[date] for date in seq_dates]
            target_image = self.images[target_date]

            input_seq_bands = np.array(sequence, dtype=np.float32)
            target_fire_band = np.array(target_image[22, :, :], dtype=np.float32)

            # --- Inspect unique values in target_fire_band ---
            current_unique_values = np.unique(target_fire_band[~np.isnan(target_fire_band)]) # Exclude NaN when finding unique values
            unique_fire_band_values.update(current_unique_values)
            #print(f"Unique values in active fire band for {target_date}: {current_unique_values}")
            #print(f"Min/Max values in active fire band for {target_date}: Min={np.nanmin(target_fire_band) if np.any(~np.isnan(target_fire_band)) else 'NaN'}, Max={np.nanmax(target_fire_band) if np.any(~np.isnan(target_fire_band)) else 'NaN'}")


            if self.image_height is None:
                self.image_height = target_fire_band.shape[0]
                self.image_width = target_fire_band.shape[1]

            for row in range(target_fire_band.shape[0]):
                for col in range(target_fire_band.shape[1]):
                    feature_vector = input_seq_bands[:, :, row, col].flatten()
                    target_value = target_fire_band[row, col]

                    # --- Create DIRECT VALUE target label ---
                    if np.isnan(target_value):
                        direct_value_target = 0.0  # Set target to 0.0 when NaN - for regression
                        nan_target_count += 1 # Increment NaN counter
                    else:
                        direct_value_target = target_value # Use direct active fire band value
                        non_nan_target_count += 1 # Increment non-NaN counter

                    #print(f"  Target Value (from band 22): {target_value}, Direct Value Target: {direct_value_target}") # DEBUG PRINT

                    self.input_features.append(feature_vector)
                    self.target_values.append(direct_value_target) # Append DIRECT VALUE target (float)
                    #print(f"  Appended target_values length: {len(self.target_values)}") # DEBUG PRINT


        #print(f"\n--- Overall Unique Values in Active Fire Band (excluding NaN) - {self.dataset_name} set ---") # Dataset name in print
        #print(f"Unique numerical values found in active fire band across dataset: {sorted(list(unique_fire_band_values))}")


        # --- Distribution of DIRECT VALUE targets ---
        target_value_stats = {
            "min": np.min(self.target_values),
            "max": np.max(self.target_values),
            "mean": np.mean(self.target_values),
            "median": np.median(self.target_values),
            "std": np.std(self.target_values),
            "zero_value_count": np.sum(self.target_values == 0.0), # Count of pixels with target 0.0 (NaNs in original)
            "non_zero_value_count": np.sum(self.target_values != 0.0), # Count of pixels with non-zero targets (original values)
            "nan_target_pixel_count": nan_target_count, # Count of original NaN target pixels
            "non_nan_target_pixel_count": non_nan_target_count # Count of original non-NaN target pixels
        }

        # Calculate NaN vs Non-NaN ratio
        total_pixels = nan_target_count + non_nan_target_count
        nan_ratio = (nan_target_count / total_pixels) * 100 if total_pixels > 0 else 0
        non_nan_ratio = (non_nan_target_count / total_pixels) * 100 if total_pixels > 0 else 0
        target_value_stats["nan_ratio_percent"] = nan_ratio
        target_value_stats["non_nan_ratio_percent"] = non_nan_ratio


        print(f"\n--- Distribution of DIRECT VALUE target values - {self.dataset_name} set: ---") 
        for stat, value in target_value_stats.items():
            print(f"{stat}: {value}")


        if not valid_sequences_exist:
            raise ValueError("No valid sequences created. Potential Data Error.")

        self.input_features = np.array(self.input_features)
        self.target_values = np.array(self.target_values).astype(np.float32)


    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        return self.input_features[idx], self.target_values[idx]


def load_images_from_directory(base_dir, years):
    all_images = {}
    target_size = None
    bands = None

    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.exists(year_dir):
            print(f"Directory not found: {year_dir}")
            continue

        fire_id_dirs = [d for d in os.listdir(year_dir) if os.path.isdir(os.path.join(year_dir, d))]
        if not fire_id_dirs:
            print(f"No fire_id directory found in: {year_dir}")
            continue

        fire_id = fire_id_dirs[0]
        fire_id_dir = os.path.join(year_dir, fire_id)

        for filename in os.listdir(fire_id_dir):
            if filename.endswith(".tif"):
                date_str = filename[:-4]
                filepath = os.path.join(fire_id_dir, filename)
                try:
                    with rasterio.open(filepath) as src:
                        image = src.read().astype(np.float32)

                        if np.isnan(np.min(image)) or np.isnan(np.max(image)):
                            print(f"WARNING: NaN found for {filename}!")

                        if target_size is None:
                            target_size = (image.shape[1], image.shape[2])
                        if bands is None:
                            bands = image.shape[0]
                        if (image.shape[1], image.shape[2]) != target_size:
                            resized_image = np.zeros((image.shape[0], target_size[0], target_size[1]), dtype=np.float32)
                            for i in range(image.shape[0]):
                                resized_image[i] = cv2.resize(image[i], dsize=(target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                            image = resized_image

                        all_images[date_str] = image

                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue

    if not all_images:
        print("Warning: No valid images loaded from directory.")

    return all_images, target_size, bands


def train_xgboost_model(train_dataset, val_dataset, test_dataset, params, num_rounds=100):
    """
    Trains the XGBoost model for DIRECT VALUE REGRESSION.

    Args:
        train_dataset (WildfireXGBoostDataset): Training dataset.
        val_dataset (WildfireXGBoostDataset): Validation dataset.
        test_dataset (WildfireXGBoostDataset): Test dataset.
        params (dict): XGBoost parameters.
        num_rounds (int): Number of boosting rounds.

    Returns:
        xgboost.Booster: Trained XGBoost model.
        dict: Training history (loss per round).
    """

    X_train = train_dataset.input_features
    y_train = train_dataset.target_values
    X_val = val_dataset.input_features
    y_val = val_dataset.target_values
    X_test = test_dataset.input_features
    y_test = test_dataset.target_values

    print("Data type of y_train:", y_train.dtype)  


    dtrain = xgb.DMatrix(X_train, label=y_train,  enable_categorical=True, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True, missing=np.nan)

    evals = [(dtrain, 'train'), (dval, 'eval')] 

    # Train the model directly, capturing history in `evals_result`
    evals_result = {}
    model = xgb.train(params, dtrain, num_rounds,
                      evals=evals, 
                      evals_result=evals_result, 
                      callbacks=[xgb.callback.EarlyStopping(rounds=10, save_best=True)]) # Early stopping

    history = evals_result 


    return model, history, dtest, y_test


def evaluate_xgboost_model(model, test_dataset, dtest, y_test, history): # Pass history
    """
    Evaluates the trained XGBoost model for DIRECT VALUE REGRESSION.

    Args:
        model (xgboost.Booster): Trained XGBoost model.
        test_dataset (WildfireXGBoostDataset): Test dataset.
        dtest (xgboost.DMatrix): DMatrix for test data
        y_test (np.array): True target values for test data
        history (dict): Training history

    Returns:
        dict: Evaluation metrics (RMSE, MAE).
    """

    y_pred_values = model.predict(dtest) 

    test_rmse = mean_squared_error(y_test, y_pred_values, squared=False) 
    test_mae = mean_absolute_error(y_test, y_pred_values) 
    train_rmse_history = history['train']['rmse'] 
    test_rmse_history = history['eval']['rmse'] 
    test_rmse_final = test_rmse_history[-1] 


    print(f"Test RMSE: {test_rmse:.4f}") 
    print(f"Test MAE: {test_mae:.4f}")   
    print(f"Final Validation RMSE from training: {test_rmse_final:.4f}") 

    return {"test_rmse": test_rmse, "test_mae": test_mae, "validation_rmse": test_rmse_final, "train_rmse_history": train_rmse_history, "validation_rmse_history": test_rmse_history} 


if __name__ == '__main__':
    base_dir = "/s/lovelace/h/nobackup/sangmi/hackathon"
    sequence_length = 2
    batch_size = 32

    train_years = [2018, 2020]
    val_years = [2019]
    test_years = [2021]

    print("Loading training data...")
    train_images, target_shape, bands = load_images_from_directory(base_dir, train_years)
    val_images, _, _ = load_images_from_directory(base_dir, val_years)
    test_images, _, _ = load_images_from_directory(base_dir, test_years)


    if train_images and val_images and test_images:
        train_dataset = WildfireXGBoostDataset(train_images, sequence_length, bands, dataset_name="Training") 
        val_dataset = WildfireXGBoostDataset(val_images, sequence_length, bands, dataset_name="Validation") 
        test_dataset = WildfireXGBoostDataset(test_images, sequence_length, bands, dataset_name="Test") 

        print("\n--- Training Data Portion (XGBoost) ---")
        print(f"Number of training image dates loaded: {len(train_images)}")
        print(f"Target shape of images: {target_shape}")
        print(f"Number of bands per image: {bands}")
        print(f"Sequence length for training: {sequence_length} days")
        print(f"Number of training samples (pixels): {len(train_dataset)}")

        # --- Print Feature-Label Mapping for the FIRST SAMPLE ---
        first_sample_features, first_sample_target = train_dataset[0]
        print("\n--- Feature-Label Mapping for the FIRST TRAINING SAMPLE ---")
        print("Feature Vector (first 20 values, flattened sequence of input bands):") 
        print(first_sample_features[:161])
        print(f"Feature Vector Shape: {first_sample_features.shape}")
        print(f"Target Value (Active Fire Band Value for target date, same pixel): {first_sample_target}")
        print("--- End Feature-Label Mapping ---")


        # XGBoost parameters - Simplified for Regression
                # XGBoost parameters - Tuned for NaN targets and Imbalance 
        xgb_params = {
            'objective': 'reg:squarederror',  # Regression objective (RMSE)
            'eval_metric': ['rmse', 'mae'], # Evaluate with RMSE and MAE
            'eta': 0.03,         # Even lower learning rate - more conservative updates
            'max_depth': 4,       # Further reduced max depth - simpler trees
            'subsample': 0.6,     # Further reduced subsample - more robust to noise
            'colsample_bytree': 0.6, # Further reduced colsample_bytree - more robust to noise
            'lambda': 1.5,        # Increased L2 regularization - stronger penalty for complexity
            'alpha': 0.2,         # Increased L1 regularization - encourage feature selection
            'min_child_weight': 5,  # Increased min_child_weight - more conservative tree splits
            'tree_method': 'hist',  # Use 'hist' for faster training with large datasets
            'seed': 42,
            'nthread': -1,
            'handle_unknown': 'use_na'
        }
        print("XGBoost parameters being used:", xgb_params) 
        trained_model, history, dtest, y_test = train_xgboost_model(train_dataset, val_dataset, test_dataset, xgb_params, num_rounds=100)

        print("\nEvaluating XGBoost model...")
        evaluation_metrics = evaluate_xgboost_model(trained_model, test_dataset, dtest, y_test, history) 


        print("\nTraining History (Validation RMSE):")
        print(history['eval']['rmse']) 
        print("\nEvaluation Metrics:")
        print(evaluation_metrics)


    else:
        print("Data loading failed. Model not trained or evaluated.")
