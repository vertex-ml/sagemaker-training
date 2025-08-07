#!/usr/bin/env python3

import os
import sys
import json
import time
from typing import Dict, Any, Optional

from .aws_auth import AWSAuthenticator
from .sagemaker_client import SageMakerClient
from .validators import InputValidator
from .utils import setup_logging, get_github_output, set_github_output


def main():
    logger = setup_logging()
    logger.info("Starting SageMaker Training Action")
    
    try:
        # Initialize components
        validator = InputValidator()
        authenticator = AWSAuthenticator()
        
        # Get and validate inputs
        inputs = get_action_inputs()
        validation_result = validator.validate_inputs(inputs)
        
        if not validation_result.is_valid:
            logger.error(f"Input validation failed: {validation_result.errors}")
            sys.exit(1)
        
        logger.info("Input validation passed")
        
        # Authenticate with AWS
        session = authenticator.get_aws_session(inputs)
        
        # Initialize SageMaker client
        sagemaker_client = SageMakerClient(session, inputs['aws-region'])
        
        # Create training job configuration
        training_config = build_training_config(inputs)
        logger.info(f"Training job configuration: {json.dumps(training_config, indent=2)}")
        
        # Submit training job
        job_name = training_config['TrainingJobName']
        logger.info(f"Submitting training job: {job_name}")
        
        response = sagemaker_client.create_training_job(training_config)
        job_arn = response['TrainingJobArn']
        
        logger.info(f"Training job submitted successfully. ARN: {job_arn}")
        
        # Set initial outputs
        set_github_output('job-name', job_name)
        set_github_output('job-arn', job_arn)
        set_github_output('training-image', training_config['AlgorithmSpecification']['TrainingImage'])
        set_github_output('training-job-definition', json.dumps(training_config))
        
        # Wait for completion if requested
        if inputs.get('wait-for-completion', 'true').lower() == 'true':
            logger.info("Waiting for training job completion...")
            
            final_status = sagemaker_client.wait_for_training_job_completion(
                job_name, 
                check_interval=int(inputs.get('check-interval', 60))
            )
            
            # Get final job details
            job_details = sagemaker_client.describe_training_job(job_name)
            
            # Set final outputs
            set_github_output('job-status', final_status)
            
            if 'ModelArtifacts' in job_details and 'S3ModelArtifacts' in job_details['ModelArtifacts']:
                set_github_output('model-artifacts', job_details['ModelArtifacts']['S3ModelArtifacts'])
            
            logger.info(f"Training job completed with status: {final_status}")
            
            if final_status != 'Completed':
                logger.error(f"Training job failed with status: {final_status}")
                if 'FailureReason' in job_details:
                    logger.error(f"Failure reason: {job_details['FailureReason']}")
                sys.exit(1)
        else:
            set_github_output('job-status', 'InProgress')
            logger.info("Training job submitted. Not waiting for completion.")
        
        logger.info("SageMaker Training Action completed successfully")
        
    except Exception as e:
        logger.error(f"Action failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


def get_action_inputs() -> Dict[str, Any]:
    inputs = {}
    
    # Define all possible inputs from action.yml
    input_mappings = {
        'aws-access-key-id': 'INPUT_AWS_ACCESS_KEY_ID',
        'aws-secret-access-key': 'INPUT_AWS_SECRET_ACCESS_KEY',
        'aws-session-token': 'INPUT_AWS_SESSION_TOKEN',
        'aws-region': 'INPUT_AWS_REGION',
        'role-to-assume': 'INPUT_ROLE_TO_ASSUME',
        'job-name': 'INPUT_JOB_NAME',
        'algorithm-specification': 'INPUT_ALGORITHM_SPECIFICATION',
        'role-arn': 'INPUT_ROLE_ARN',
        'instance-type': 'INPUT_INSTANCE_TYPE',
        'instance-count': 'INPUT_INSTANCE_COUNT',
        'volume-size': 'INPUT_VOLUME_SIZE',
        'max-runtime': 'INPUT_MAX_RUNTIME',
        'input-data-config': 'INPUT_INPUT_DATA_CONFIG',
        'output-data-config': 'INPUT_OUTPUT_DATA_CONFIG',
        'hyperparameters': 'INPUT_HYPERPARAMETERS',
        'environment': 'INPUT_ENVIRONMENT',
        'vpc-config': 'INPUT_VPC_CONFIG',
        'tags': 'INPUT_TAGS',
        'wait-for-completion': 'INPUT_WAIT_FOR_COMPLETION',
        'check-interval': 'INPUT_CHECK_INTERVAL'
    }
    
    for action_input, env_var in input_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            inputs[action_input] = value
    
    return inputs


def build_training_config(inputs: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        'TrainingJobName': inputs['job-name'],
        'RoleArn': inputs['role-arn'],
        'AlgorithmSpecification': {
            'TrainingImage': inputs['algorithm-specification'],
            'TrainingInputMode': 'File'
        },
        'InputDataConfig': json.loads(inputs['input-data-config']),
        'OutputDataConfig': json.loads(inputs['output-data-config']),
        'ResourceConfig': {
            'InstanceType': inputs.get('instance-type', 'ml.m5.large'),
            'InstanceCount': int(inputs.get('instance-count', 1)),
            'VolumeSizeInGB': int(inputs.get('volume-size', 30))
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': int(inputs.get('max-runtime', 86400))
        }
    }
    
    # Add optional configurations
    if inputs.get('hyperparameters'):
        hyperparameters = json.loads(inputs['hyperparameters'])
        if hyperparameters:
            config['HyperParameters'] = {k: str(v) for k, v in hyperparameters.items()}
    
    if inputs.get('environment'):
        environment = json.loads(inputs['environment'])
        if environment:
            config['Environment'] = {k: str(v) for k, v in environment.items()}
    
    if inputs.get('vpc-config'):
        vpc_config = json.loads(inputs['vpc-config'])
        if vpc_config:
            config['VpcConfig'] = vpc_config
    
    if inputs.get('tags'):
        tags = json.loads(inputs['tags'])
        if tags:
            config['Tags'] = [{'Key': k, 'Value': str(v)} for k, v in tags.items()]
    
    return config


if __name__ == '__main__':
    main()