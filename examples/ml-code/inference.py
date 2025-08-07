#!/usr/bin/env python3
"""
Custom Inference Handler for SageMaker Endpoints
This script handles model loading and prediction requests
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold model artifacts
model = None
scaler = None
label_encoder = None
feature_names = None
model_metadata = None

def model_fn(model_dir):
    """
    Load model and preprocessing artifacts from model directory
    This function is called once when the model server starts
    """
    global model, scaler, label_encoder, feature_names, model_metadata
    
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # Load metadata first
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Loaded model metadata: {model_metadata['model_name']}")
        
        # Find and load the main model file
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') and 'scaler' not in f and 'encoder' not in f]
        
        if not model_files:
            raise FileNotFoundError("No model file (.joblib) found in model directory")
        
        model_path = os.path.join(model_dir, model_files[0])
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load scaler if available
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")
        
        # Load label encoder if available
        encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            logger.info("Loaded label encoder")
        
        # Load feature names if available
        feature_path = os.path.join(model_dir, 'feature_names.json')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        logger.info("Model loading completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """
    Parse and preprocess input data for predictions
    Supports multiple input formats: CSV, JSON
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    try:
        if request_content_type == 'text/csv':
            # Handle CSV input
            logger.info("Processing CSV input")
            
            # Read CSV data
            df = pd.read_csv(StringIO(request_body))
            logger.info(f"Parsed CSV with shape: {df.shape}")
            
            # Validate feature names if available
            if feature_names and list(df.columns) != feature_names:
                logger.warning(f"Input features {list(df.columns)} don't match training features {feature_names}")
                # Try to reorder columns to match training data
                if all(col in df.columns for col in feature_names):
                    df = df[feature_names]
                    logger.info("Reordered columns to match training data")
            
            return df
            
        elif request_content_type == 'application/json':
            # Handle JSON input
            logger.info("Processing JSON input")
            
            input_data = json.loads(request_body)
            
            if isinstance(input_data, dict):
                # Single instance as dictionary
                if 'instances' in input_data:
                    # Batch prediction format: {"instances": [{"feature1": value1, ...}, ...]}
                    instances = input_data['instances']
                    df = pd.DataFrame(instances)
                elif 'data' in input_data:
                    # Alternative format: {"data": [[value1, value2, ...], ...]}
                    data = input_data['data']
                    if feature_names:
                        df = pd.DataFrame(data, columns=feature_names)
                    else:
                        df = pd.DataFrame(data)
                else:
                    # Single instance: {"feature1": value1, "feature2": value2, ...}
                    df = pd.DataFrame([input_data])
            
            elif isinstance(input_data, list):
                # Multiple instances as list of dictionaries or list of lists
                if all(isinstance(item, dict) for item in input_data):
                    # List of dictionaries: [{"feature1": value1, ...}, ...]
                    df = pd.DataFrame(input_data)
                else:
                    # List of lists: [[value1, value2, ...], ...]
                    if feature_names:
                        df = pd.DataFrame(input_data, columns=feature_names)
                    else:
                        df = pd.DataFrame(input_data)
            
            else:
                raise ValueError("Invalid JSON input format")
            
            logger.info(f"Parsed JSON with shape: {df.shape}")
            return df
            
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
            
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise

def predict_fn(input_data, model):
    """
    Make predictions on the input data
    """
    logger.info(f"Making predictions on data with shape: {input_data.shape}")
    
    try:
        # Apply preprocessing if scaler is available
        if scaler is not None:
            logger.info("Applying feature scaling")
            input_data_scaled = pd.DataFrame(
                scaler.transform(input_data),
                columns=input_data.columns
            )
        else:
            input_data_scaled = input_data
        
        # Make predictions
        predictions = model.predict(input_data_scaled)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data_scaled)
            logger.info("Generated prediction probabilities")
        
        # Convert predictions back to original labels if label encoder exists
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)
            logger.info("Converted predictions to original labels")
        
        # Prepare output
        result = {
            'predictions': predictions.tolist()
        }
        
        if probabilities is not None:
            result['probabilities'] = probabilities.tolist()
            
            # Add class names if available
            if label_encoder is not None:
                result['classes'] = label_encoder.classes_.tolist()
        
        # Add metadata
        if model_metadata:
            result['model_info'] = {
                'model_name': model_metadata.get('model_name', 'unknown'),
                'model_type': model_metadata.get('model_type', 'unknown'),
                'version': model_metadata.get('github_sha', 'unknown')[:8]
            }
        
        logger.info("Prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction, accept):
    """
    Format the prediction output based on the requested accept type
    """
    logger.info(f"Formatting output for accept type: {accept}")
    
    try:
        if accept == 'application/json':
            return json.dumps(prediction), accept
            
        elif accept == 'text/csv':
            # Convert to CSV format - just predictions
            predictions_df = pd.DataFrame({
                'prediction': prediction['predictions']
            })
            
            # Add probabilities if available
            if 'probabilities' in prediction:
                probs = prediction['probabilities']
                if 'classes' in prediction:
                    # Named probability columns
                    classes = prediction['classes']
                    for i, class_name in enumerate(classes):
                        predictions_df[f'prob_{class_name}'] = [prob[i] for prob in probs]
                else:
                    # Numbered probability columns
                    for i in range(len(probs[0])):
                        predictions_df[f'prob_{i}'] = [prob[i] for prob in probs]
            
            return predictions_df.to_csv(index=False), accept
            
        elif accept == 'text/plain':
            # Simple text format - just predictions
            predictions = prediction['predictions']
            return '\n'.join(map(str, predictions)), accept
            
        else:
            # Default to JSON
            logger.warning(f"Unsupported accept type {accept}, defaulting to JSON")
            return json.dumps(prediction), 'application/json'
            
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        # Return error in JSON format
        error_response = {
            'error': f"Error formatting output: {str(e)}",
            'predictions': prediction.get('predictions', [])
        }
        return json.dumps(error_response), 'application/json'

# Health check endpoint (optional)
def ping():
    """
    Health check function - returns 200 if model is loaded successfully
    """
    if model is not None:
        return '', 200
    else:
        return 'Model not loaded', 500