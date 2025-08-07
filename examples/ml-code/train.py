#!/usr/bin/env python3
"""
Custom Python Training Script for SageMaker
This is an example showing how to structure your custom training code
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import boto3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Custom SageMaker Training')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    
    # Model hyperparameters
    parser.add_argument('--model-type', type=str, default='random_forest', 
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'])
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=42)
    
    # Data processing parameters
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--scale-features', action='store_true', help='Apply feature scaling')
    parser.add_argument('--cross-validation', type=int, default=5, help='Number of CV folds')
    
    # Custom parameters
    parser.add_argument('--model-name', type=str, default='custom_model')
    parser.add_argument('--target-column', type=str, default='target', help='Name of target column')
    
    return parser.parse_args()

def load_data(data_path, target_column='target'):
    """Load and combine training data from multiple files"""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Handle both directory and single file cases
    data_files = []
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            if file.endswith(('.csv', '.tsv', '.txt')):
                data_files.append(os.path.join(data_path, file))
    else:
        data_files = [data_path]
    
    if not data_files:
        raise ValueError("No CSV/TSV files found in the data path")
    
    # Load and concatenate all data files
    dataframes = []
    for file_path in sorted(data_files):
        logger.info(f"Reading file: {file_path}")
        try:
            # Try different separators
            if file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                df = pd.read_csv(file_path)
            
            dataframes.append(df)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    # Combine all dataframes
    combined_data = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total data shape: {combined_data.shape}")
    
    # Handle target column
    if target_column not in combined_data.columns:
        # If target column not found, assume last column is target
        target_column = combined_data.columns[-1]
        logger.info(f"Target column '{target_column}' not found, using last column: {target_column}")
    
    return combined_data, target_column

def preprocess_data(data, target_column, scale_features=False):
    """Preprocess the data for training"""
    logger.info("Starting data preprocessing...")
    
    # Remove rows with missing values
    initial_rows = len(data)
    data_clean = data.dropna()
    logger.info(f"Removed {initial_rows - len(data_clean)} rows with missing values")
    
    # Separate features and target
    X = data_clean.drop(columns=[target_column])
    y = data_clean[target_column]
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        logger.info(f"Found categorical columns: {list(categorical_columns)}")
        # Simple label encoding for categorical features
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Handle target variable if it's categorical
    label_encoder = None
    if y.dtype == 'object':
        logger.info("Target variable is categorical, applying label encoding")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Feature scaling
    scaler = None
    if scale_features:
        logger.info("Applying feature scaling...")
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    logger.info(f"Preprocessed data - Features: {X.shape}, Target: {y.shape}")
    logger.info(f"Target classes: {np.unique(y)}")
    
    return X, y, scaler, label_encoder

def create_model(model_type, hyperparams):
    """Create model based on specified type and hyperparameters"""
    logger.info(f"Creating {model_type} model...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            min_samples_split=hyperparams['min_samples_split'],
            min_samples_leaf=hyperparams['min_samples_leaf'],
            random_state=hyperparams['random_state'],
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            min_samples_split=hyperparams['min_samples_split'],
            min_samples_leaf=hyperparams['min_samples_leaf'],
            random_state=hyperparams['random_state']
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=hyperparams['random_state'],
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def train_model(model, X_train, y_train, cv_folds=5):
    """Train the model with cross-validation"""
    logger.info("Starting model training...")
    
    # Cross-validation before final training
    if cv_folds > 1:
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test, label_encoder=None):
    """Evaluate model performance"""
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Get class names for reporting
    target_names = None
    if label_encoder:
        target_names = label_encoder.classes_
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        logger.info("Top 10 Feature Importances:")
        for i, importance in enumerate(sorted(enumerate(feature_importance), 
                                            key=lambda x: x[1], reverse=True)[:10]):
            logger.info(f"Feature {importance[0]}: {importance[1]:.4f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importances': feature_importance.tolist() if feature_importance is not None else None
    }

def save_model_and_artifacts(model, model_dir, model_name, metrics, hyperparams, 
                           scaler=None, label_encoder=None, feature_names=None):
    """Save model and all related artifacts"""
    logger.info(f"Saving model artifacts to {model_dir}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the main model
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessing artifacts
    if scaler:
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    if label_encoder:
        encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
        joblib.dump(label_encoder, encoder_path)
        logger.info(f"Label encoder saved to {encoder_path}")
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'scikit_learn_version': '1.3.0',
        'hyperparameters': hyperparams,
        'metrics': metrics,
        'feature_names': feature_names,
        'has_scaler': scaler is not None,
        'has_label_encoder': label_encoder is not None,
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'github_sha': os.environ.get('GITHUB_SHA', 'unknown'),
        'training_job_name': os.environ.get('TRAINING_JOB_NAME', 'unknown')
    }
    
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {metadata_path}")
    
    # Save feature names
    if feature_names is not None:
        feature_path = os.path.join(model_dir, 'feature_names.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f)
        logger.info(f"Feature names saved to {feature_path}")

def upload_metrics_to_s3(metrics, job_name):
    """Upload training metrics to S3 for tracking"""
    try:
        s3_client = boto3.client('s3')
        bucket = os.environ.get('METRICS_BUCKET', 'my-ml-bucket')
        key = f'training-metrics/{job_name}/metrics.json'
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metrics, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Metrics uploaded to s3://{bucket}/{key}")
    except Exception as e:
        logger.warning(f"Failed to upload metrics to S3: {str(e)}")

def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting custom Python model training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load training data
        train_data, target_column = load_data(args.train, args.target_column)
        
        # Preprocess data
        X, y, scaler, label_encoder = preprocess_data(
            train_data, target_column, args.scale_features
        )
        
        # Split data if no separate validation set
        if not os.path.exists(args.validation):
            logger.info("No separate validation data, splitting training data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, 
                random_state=args.random_state, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
        else:
            logger.info("Using separate validation dataset")
            X_train, y_train = X, y
            val_data, _ = load_data(args.validation, target_column)
            X_test, y_test, _, _ = preprocess_data(val_data, target_column, args.scale_features)
        
        # Prepare hyperparameters
        hyperparams = {
            'model_type': args.model_type,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'learning_rate': args.learning_rate,
            'random_state': args.random_state,
            'scale_features': args.scale_features
        }
        
        # Create and train model
        model = create_model(args.model_type, hyperparams)
        model = train_model(model, X_train, y_train, args.cross_validation)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Save model and artifacts
        save_model_and_artifacts(
            model, args.model_dir, args.model_name, metrics, hyperparams,
            scaler, label_encoder, list(X.columns)
        )
        
        # Upload metrics (optional)
        job_name = os.environ.get('TRAINING_JOB_NAME')
        if job_name:
            upload_metrics_to_s3(metrics, job_name)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Final accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()