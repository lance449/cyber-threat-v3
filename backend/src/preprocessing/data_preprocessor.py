"""
Data Preprocessing Module for Cyber Threat Detection System
Handles the All_Network.csv dataset with comprehensive feature engineering
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Unified data preprocessing class for cyber threat detection
    Handles All_Network.csv dataset with comprehensive feature engineering
    """
    
    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize the data preprocessor
        
        Args:
            scaler_type: Type of scaler to use ("minmax" or "standard")
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoder = None
        self.imputer = None
        self.feature_names = []
        self.metadata_columns = []
        self.class_weights = {}
        
    def load_dataset(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load the All_Network.csv dataset with optional sampling
        
        Args:
            file_path: Path to the CSV file
            sample_size: If provided, sample this many rows for faster processing
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        # Load dataset with optimized settings
        if sample_size:
            # Read only the first few rows to get column names
            df_sample = pd.read_csv(file_path, nrows=5)
            columns = df_sample.columns.tolist()

            # Count total data rows (excluding header)
            total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - 1

            # Cap requested sample to available rows to avoid negative skip sizes
            effective_sample = max(0, min(int(sample_size), int(total_rows)))

            if effective_sample <= 0:
                # Fallback: read minimal safe chunk
                logger.warning("Requested sample size is 0 after capping; loading empty dataframe with columns only")
                df = pd.read_csv(file_path, nrows=0)
            elif effective_sample >= total_rows:
                # No sampling needed; read full file
                logger.info(
                    f"Requested sample_size {sample_size} >= total_rows {total_rows}; loading full dataset"
                )
                df = pd.read_csv(file_path, low_memory=False)
            else:
                # Randomly skip rows to keep exactly 'effective_sample' rows
                n_to_skip = total_rows - effective_sample
                # Use numpy for efficient random selection of rows to skip (1..total_rows)
                skip_rows = sorted(
                    np.random.choice(range(1, total_rows + 1), n_to_skip, replace=False)
                )
                df = pd.read_csv(file_path, skiprows=skip_rows, low_memory=False)
                logger.info(
                    f"Sampled dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns (from {total_rows} total)"
                )
        else:
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and infinite values
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning dataset...")
        
        # Store original shape
        original_shape = df.shape
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Log cleaning results
        logger.info(f"Data cleaning completed:")
        logger.info(f"  - Removed {original_shape[0] - df.shape[0]} rows")
        logger.info(f"  - Removed {original_shape[1] - df.shape[1]} columns")
        
        return df
    
    def extract_metadata(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract metadata columns for later use
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, metadata_df)
        """
        # Define metadata columns
        metadata_cols = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"]
        present_metadata = [col for col in metadata_cols if col in df.columns]
        
        # Extract metadata
        metadata_df = df[present_metadata].copy() if present_metadata else pd.DataFrame(index=df.index)
        
        # Remove metadata from features
        features_df = df.drop(columns=present_metadata, errors='ignore')
        
        self.metadata_columns = present_metadata
        
        logger.info(f"Extracted {len(present_metadata)} metadata columns")
        
        return features_df, metadata_df
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the label column for classification
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded labels
        """
        # Find label column
        label_col = None
        for col in df.columns:
            if col.lower() == "label":
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("No 'Label' column found in dataset")
        
        # Encode labels
        labels_raw = df[label_col].astype(str).fillna("UNKNOWN")
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels_raw)
        
        # Replace original label with encoded version
        df = df.drop(columns=[label_col])
        df["_label_encoded"] = labels_encoded
        
        # Log class distribution
        unique_labels, counts = np.unique(labels_encoded, return_counts=True)
        logger.info(f"Label encoding completed. Classes: {list(self.label_encoder.classes_)}")
        logger.info("Class distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(df)) * 100
            logger.info(f"  Class {self.label_encoder.classes_[label]}: {count:,} samples ({percentage:.1f}%)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for better detection
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering additional features...")
        
        # Create copy to avoid modifying original
        df_eng = df.copy()
        
        # Network flow features
        if 'Flow Duration' in df_eng.columns:
            df_eng['flow_duration_log'] = np.log1p(df_eng['Flow Duration'].abs())
            df_eng['flow_duration_std'] = (df_eng['Flow Duration'] - df_eng['Flow Duration'].mean()) / df_eng['Flow Duration'].std()
        
        # Packet size features
        if 'Packet Length Mean' in df_eng.columns:
            df_eng['packet_size_ratio'] = df_eng['Packet Length Mean'] / (df_eng['Packet Length Mean'].max() + 1e-8)
            df_eng['packet_size_std'] = (df_eng['Packet Length Mean'] - df_eng['Packet Length Mean'].mean()) / df_eng['Packet Length Mean'].std()
        
        # Flow rate features
        if 'Flow Bytes/s' in df_eng.columns:
            df_eng['bytes_per_sec_log'] = np.log1p(df_eng['Flow Bytes/s'].abs())
            df_eng['bytes_per_sec_std'] = (df_eng['Flow Bytes/s'] - df_eng['Flow Bytes/s'].mean()) / df_eng['Flow Bytes/s'].std()
        
        # Flag-based features (vectorized)
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
                    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']
        present_flags = [col for col in flag_cols if col in df_eng.columns]
        
        if present_flags:
            # Ensure numeric and non-negative
            flags_mat = df_eng[present_flags].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
            totals = flags_mat.sum(axis=1)
            df_eng['total_flags'] = totals
            # Probability distribution over flags per row
            denom = (totals + 1e-8).reshape(-1, 1)
            p = flags_mat / denom
            # Entropy: -sum(p * log2(p)) with safe log
            entropy = -np.sum(p * np.log2(np.clip(p, 1e-12, 1.0)), axis=1)
            df_eng['flag_entropy'] = entropy.astype(np.float32)
        
        # IAT (Inter Arrival Time) features
        iat_cols = ['Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean']
        present_iat = [col for col in iat_cols if col in df_eng.columns]
        
        if present_iat:
            df_eng['iat_mean'] = df_eng[present_iat].mean(axis=1)
            df_eng['iat_std'] = df_eng[present_iat].std(axis=1)
            df_eng['iat_ratio'] = df_eng['iat_std'] / (df_eng['iat_mean'] + 1e-8)
        
        # Protocol-specific features
        if 'Protocol' in df_eng.columns:
            df_eng['protocol_encoded'] = pd.Categorical(df_eng['Protocol']).codes
        
        # Port-based features
        if 'Src Port' in df_eng.columns and 'Dst Port' in df_eng.columns:
            df_eng['port_diff'] = abs(df_eng['Src Port'] - df_eng['Dst Port'])
            df_eng['is_well_known_src'] = df_eng['Src Port'] <= 1024
            df_eng['is_well_known_dst'] = df_eng['Dst Port'] <= 1024
        
        logger.info(f"Feature engineering completed. Total features: {df_eng.shape[1]}")
        
        return df_eng
    
    def handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle categorical features by encoding them
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Identify categorical columns
        categorical_cols = []
        
        for col in df.columns:
            if col in ['_label_encoded']:  # Skip label column
                continue
            if df[col].dtype == object:
                # Check if it's a hash-like column (long string)
                sample_value = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
                if len(sample_value) > 20:  # Likely a hash
                    categorical_cols.append(col)
                else:
                    # For other object columns, check unique values
                    nunique = df[col].nunique(dropna=True)
                    if nunique <= 100:  # Increased threshold
                        categorical_cols.append(col)
        
        # Encode categorical features
        for col in categorical_cols:
            try:
                # Fill NaN values
                df[col] = df[col].fillna("___NA___").astype(str)
                
                # Factorize (ordinal encoding)
                codes, uniques = pd.factorize(df[col], sort=True)
                df[f"{col}_encoded"] = codes.astype(np.int32)
                
                # Drop original column
                df = df.drop(columns=[col])
                
                logger.info(f"Encoded categorical column: {col} -> {col}_encoded")
                
            except Exception as e:
                logger.warning(f"Failed to encode column {col}: {e}")
                # Drop problematic column
                df = df.drop(columns=[col])
                logger.info(f"Dropped problematic column: {col}")
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features with memory optimization
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        # Identify numerical columns (excluding label)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != '_label_encoded']
        
        if not numerical_cols:
            logger.warning("No numerical features found for scaling")
            return df
        
        # Process in chunks to save memory
        chunk_size = 100000  # Process 100k rows at a time
        total_rows = len(df)
        
        logger.info(f"Processing {total_rows} rows in chunks of {chunk_size}")
        
        # Initialize scaler and imputer
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.imputer = SimpleImputer(strategy='median')
        
        # Process chunks for fitting
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx][numerical_cols]
            
            # Convert to numeric and handle missing values
            chunk_numeric = chunk.apply(pd.to_numeric, errors='coerce')
            
            # Fit imputer and scaler on first chunk
            if start_idx == 0:
                self.imputer.fit(chunk_numeric)
                chunk_imputed = self.imputer.transform(chunk_numeric)
                self.scaler.fit(chunk_imputed)
            else:
                # Partial fit for subsequent chunks
                chunk_imputed = self.imputer.transform(chunk_numeric)
                self.scaler.partial_fit(chunk_imputed)
            
            logger.info(f"Processed chunk {start_idx//chunk_size + 1}/{(total_rows-1)//chunk_size + 1}")
        
        # Now transform all data
        df_scaled = df.copy()
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx][numerical_cols]
            
            # Convert to numeric and handle missing values
            chunk_numeric = chunk.apply(pd.to_numeric, errors='coerce')
            chunk_imputed = self.imputer.transform(chunk_numeric)
            chunk_scaled = self.scaler.transform(chunk_imputed)
            
            # Update DataFrame with scaled features (use float32 to save memory)
            # Convert DataFrame to float32 first to avoid dtype warnings
            for i, col in enumerate(numerical_cols):
                # Ensure the column is float32 before assignment
                if df_scaled[col].dtype != np.float32:
                    df_scaled[col] = df_scaled[col].astype(np.float32)
                df_scaled.loc[df_scaled.index[start_idx:end_idx], col] = chunk_scaled[:, i].astype(np.float32)
        
        # Store feature names
        self.feature_names = [col for col in df_scaled.columns if col != '_label_encoded']
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features using {self.scaler_type} scaler")
        
        return df_scaled
    
    def final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleanup to ensure all columns are numerical
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with only numerical columns
        """
        # Check for any remaining object columns
        object_cols = df.select_dtypes(include=[object]).columns.tolist()
        object_cols = [col for col in object_cols if col != '_label_encoded']
        
        if object_cols:
            logger.warning(f"Found {len(object_cols)} remaining object columns: {object_cols}")
            logger.info("Dropping remaining object columns to ensure numerical data")
            df = df.drop(columns=object_cols)
        
        # Ensure all remaining columns are numerical
        for col in df.columns:
            if col != '_label_encoded':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert column {col} to numeric, dropping it")
                    df = df.drop(columns=[col])
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        logger.info(f"Final cleanup completed. Final shape: {df.shape}")
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42, 
                   balance_classes: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed
            balance_classes: Whether to ensure balanced class distribution
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and labels
        X = df.drop(columns=['_label_encoded'])
        y = df['_label_encoded']
        
        # Log class distribution before split
        unique_labels, counts = np.unique(y, return_counts=True)
        logger.info("Class distribution before split:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(y)) * 100
            logger.info(f"  Class {label}: {count:,} samples ({percentage:.1f}%)")
        
        # Ensure minimum samples per class in test set
        min_test_samples = max(100, int(len(y) * test_size * 0.01))  # At least 1% of test size per class
        
        # Split data
        if balance_classes:
            # Use stratified split to ensure each class appears in both sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Verify class distribution in test set
            test_labels, test_counts = np.unique(y_test, return_counts=True)
            logger.info("Class distribution in test set:")
            for label, count in zip(test_labels, test_counts):
                if count < min_test_samples:
                    logger.warning(f"  Class {label}: {count} samples (below minimum {min_test_samples})")
                else:
                    logger.info(f"  Class {label}: {count} samples")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        logger.info(f"Data split completed:")
        logger.info(f"  - Training set: {X_train.shape[0]} samples")
        logger.info(f"  - Testing set: {X_test.shape[0]} samples")
        logger.info(f"  - Classes in training: {len(np.unique(y_train))}")
        logger.info(f"  - Classes in testing: {len(np.unique(y_test))}")
        
        return X_train, X_test, y_train, y_test
    
    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        try:
            # Compute balanced class weights
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            # Convert numpy types to Python native types for JSON compatibility
            class_weights = {int(class_idx): float(weight) for class_idx, weight in zip(classes, weights)}
            
            logger.info("Class weights computed:")
            for class_idx, weight in class_weights.items():
                logger.info(f"  Class {class_idx}: weight {weight:.3f}")
            
            self.class_weights = class_weights
            return class_weights
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")
            return {}
    
    def _resample_minority_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resample minority classes (especially Benign) to improve model detection
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Resampled (X, y) with better class balance
        """
        try:
            from sklearn.utils import resample
            
            # Get class distribution
            class_counts = y.value_counts()
            max_count = class_counts.max()
            
            logger.info("Resampling minority classes for better Benign detection...")
            logger.info(f"Original distribution: {dict(class_counts)}")
            
            # Target: ensure each class has at least 20% of the majority class
            target_min_samples = max(int(max_count * 0.2), 100)  # At least 100 samples per class
            
            resampled_samples = []
            resampled_labels = []
            
            for class_label in y.unique():
                class_mask = y == class_label
                class_X = X[class_mask]
                class_y = y[class_mask]
                
                if len(class_X) < target_min_samples:
                    # Upsample minority classes
                    logger.info(f"Upsampling class {self.label_encoder.classes_[class_label]} from {len(class_X)} to {target_min_samples} samples")
                    class_X_resampled, class_y_resampled = resample(
                        class_X, class_y, n_samples=target_min_samples, random_state=42, replace=True
                    )
                else:
                    # Keep majority classes as is
                    class_X_resampled, class_y_resampled = class_X, class_y
                
                resampled_samples.append(class_X_resampled)
                resampled_labels.append(class_y_resampled)
            
            # Combine all resampled data
            X_resampled = pd.concat(resampled_samples, ignore_index=True)
            y_resampled = pd.concat(resampled_labels, ignore_index=True)
            
            logger.info(f"Resampled distribution: {dict(y_resampled.value_counts())}")
            logger.info(f"Dataset size: {X.shape[0]} -> {X_resampled.shape[0]} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"Could not resample minority classes: {e}")
            return X, y
    
    def preprocess(self, input_path: str, output_dir: str, test_size: float = 0.3, sample_size: Optional[int] = None) -> Dict:
        """
        Complete preprocessing pipeline with memory optimization
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save processed data
            test_size: Proportion of data for testing
            sample_size: If provided, sample this many rows for faster processing
            
        Returns:
            Dictionary with preprocessing results and file paths
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load and clean data (with optional sampling)
        df = self.load_dataset(input_path, sample_size=sample_size)
        df = self.clean_data(df)
        
        # Extract metadata
        features_df, metadata_df = self.extract_metadata(df)
        
        # Encode labels
        features_df = self.encode_labels(features_df)
        
        # Engineer features
        features_df = self.engineer_features(features_df)
        
        # Handle categorical features
        features_df = self.handle_categorical_features(features_df)
        
        # Final cleanup before scaling
        features_df = self.final_cleanup(features_df)
        
        # Scale features (memory optimized)
        features_df = self.scale_features(features_df)
        
        # Get class weights for imbalanced datasets
        self.get_class_weights(features_df['_label_encoded'])
        
        # Split data with class balancing
        X_train, X_test, y_train, y_test = self.split_data(features_df, test_size, balance_classes=True)
        
        # Apply resampling to improve minority class representation (especially Benign)
        # Check for severe imbalance and automatically resample
        class_counts = y_train.value_counts()
        if len(class_counts) > 1:  # Check if we have multiple classes
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 5:  # Severe imbalance - auto-resample
                logger.info(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f}), applying resampling...")
                X_train, y_train = self._resample_minority_classes(X_train, y_train)
        
        # Save processed data
        results = self.save_processed_data(
            X_train, X_test, y_train, y_test, metadata_df, output_dir
        )
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return results
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          metadata_df: pd.DataFrame, output_dir: str) -> Dict:
        """
        Save processed data and artifacts
        
        Args:
            X_train, X_test, y_train, y_test: Split data
            metadata_df: Metadata DataFrame
            output_dir: Output directory
            
        Returns:
            Dictionary with file paths and metadata
        """
        # Save processed data
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
        joblib.dump(self.imputer, os.path.join(output_dir, 'imputer.joblib'))
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        # Save class weights
        if self.class_weights:
            joblib.dump(self.class_weights, os.path.join(output_dir, 'class_weights.joblib'))
        
        # Save preprocessing metadata
        preprocessing_info = {
            'feature_count': int(len(self.feature_names)),
            'scaler_type': self.scaler_type,
            'metadata_columns': self.metadata_columns,
            'label_classes': [str(cls) for cls in self.label_encoder.classes_] if self.label_encoder else [],
            'class_weights': {str(k): float(v) for k, v in self.class_weights.items()} if self.class_weights else {}
        }
        
        with open(os.path.join(output_dir, 'preprocessing_info.json'), 'w') as f:
            import json
            json.dump(preprocessing_info, f, indent=2)
        
        return {
            'X_train_path': os.path.join(output_dir, 'X_train.csv'),
            'X_test_path': os.path.join(output_dir, 'X_test.csv'),
            'y_train_path': os.path.join(output_dir, 'y_train.csv'),
            'y_test_path': os.path.join(output_dir, 'y_test.csv'),
            'metadata_path': os.path.join(output_dir, 'metadata.csv'),
            'scaler_path': os.path.join(output_dir, 'scaler.joblib'),
            'label_encoder_path': os.path.join(output_dir, 'label_encoder.joblib'),
            'imputer_path': os.path.join(output_dir, 'imputer.joblib'),
            'feature_names_path': os.path.join(output_dir, 'feature_names.txt'),
            'class_weights_path': os.path.join(output_dir, 'class_weights.joblib') if self.class_weights else None,
            'preprocessing_info': preprocessing_info
        }
