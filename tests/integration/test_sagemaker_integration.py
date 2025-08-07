import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from moto import mock_sagemaker, mock_sts
import boto3
from src.sagemaker_client import SageMakerClient
from src.aws_auth import AWSAuthenticator


@mock_sagemaker
@mock_sts
class TestSageMakerIntegration:
    
    def setup_method(self):
        # Create mock AWS session
        self.session = boto3.Session(region_name='us-east-1')
        self.sagemaker_client = SageMakerClient(self.session, 'us-east-1')
        
        # Sample training job configuration
        self.training_config = {
            'TrainingJobName': 'test-training-job',
            'RoleArn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
            'AlgorithmSpecification': {
                'TrainingImage': '123456789012.dkr.ecr.us-east-1.amazonaws.com/my-algorithm:latest',
                'TrainingInputMode': 'File'
            },
            'InputDataConfig': [{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://my-bucket/training-data/'
                    }
                }
            }],
            'OutputDataConfig': {
                'S3OutputPath': 's3://my-bucket/output/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400
            }
        }
    
    def test_create_training_job_success(self):
        # Mock successful response
        with patch.object(self.sagemaker_client.sagemaker, 'create_training_job') as mock_create:
            mock_create.return_value = {
                'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789012:training-job/test-training-job'
            }
            
            response = self.sagemaker_client.create_training_job(self.training_config)
            
            # Verify the call was made with correct parameters
            mock_create.assert_called_once_with(**self.training_config)
            
            # Verify response
            assert 'TrainingJobArn' in response
            assert 'test-training-job' in response['TrainingJobArn']
    
    def test_create_training_job_failure(self):
        # Mock failure response
        with patch.object(self.sagemaker_client.sagemaker, 'create_training_job') as mock_create:
            mock_create.side_effect = Exception("Resource limit exceeded")
            
            with pytest.raises(Exception, match="Resource limit exceeded"):
                self.sagemaker_client.create_training_job(self.training_config)
    
    def test_describe_training_job(self):
        job_name = 'test-training-job'
        mock_response = {
            'TrainingJobName': job_name,
            'TrainingJobStatus': 'InProgress',
            'TrainingJobArn': f'arn:aws:sagemaker:us-east-1:123456789012:training-job/{job_name}',
            'CreationTime': '2023-01-01T00:00:00Z'
        }
        
        with patch.object(self.sagemaker_client.sagemaker, 'describe_training_job') as mock_describe:
            mock_describe.return_value = mock_response
            
            response = self.sagemaker_client.describe_training_job(job_name)
            
            mock_describe.assert_called_once_with(TrainingJobName=job_name)
            assert response == mock_response
    
    def test_wait_for_training_job_completion_success(self):
        job_name = 'test-training-job'
        
        # Mock responses for different stages
        responses = [
            {'TrainingJobStatus': 'InProgress'},
            {'TrainingJobStatus': 'InProgress'},
            {'TrainingJobStatus': 'Completed'}
        ]
        
        with patch.object(self.sagemaker_client, 'describe_training_job') as mock_describe:
            mock_describe.side_effect = responses
            
            with patch('time.sleep'):  # Mock sleep to speed up test
                status = self.sagemaker_client.wait_for_training_job_completion(
                    job_name, check_interval=1
                )
                
                assert status == 'Completed'
                assert mock_describe.call_count == 3
    
    def test_wait_for_training_job_completion_failure(self):
        job_name = 'test-training-job'
        
        # Mock failure response
        mock_response = {
            'TrainingJobStatus': 'Failed',
            'FailureReason': 'Algorithm error'
        }
        
        with patch.object(self.sagemaker_client, 'describe_training_job') as mock_describe:
            mock_describe.return_value = mock_response
            
            status = self.sagemaker_client.wait_for_training_job_completion(
                job_name, check_interval=1
            )
            
            assert status == 'Failed'
    
    def test_wait_for_training_job_timeout(self):
        job_name = 'test-training-job'
        
        # Mock perpetual in-progress
        with patch.object(self.sagemaker_client, 'describe_training_job') as mock_describe:
            mock_describe.return_value = {'TrainingJobStatus': 'InProgress'}
            
            with patch('time.sleep'):  # Mock sleep
                with pytest.raises(TimeoutError, match="did not complete within"):
                    self.sagemaker_client.wait_for_training_job_completion(
                        job_name, check_interval=1, max_wait_time=2
                    )
    
    def test_stop_training_job(self):
        job_name = 'test-training-job'
        
        with patch.object(self.sagemaker_client.sagemaker, 'stop_training_job') as mock_stop:
            self.sagemaker_client.stop_training_job(job_name)
            
            mock_stop.assert_called_once_with(TrainingJobName=job_name)
    
    def test_list_training_jobs(self):
        mock_response = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobStatus': 'Completed',
                    'CreationTime': '2023-01-01T00:00:00Z'
                },
                {
                    'TrainingJobName': 'job-2',
                    'TrainingJobStatus': 'InProgress',
                    'CreationTime': '2023-01-02T00:00:00Z'
                }
            ]
        }
        
        with patch.object(self.sagemaker_client.sagemaker, 'list_training_jobs') as mock_list:
            mock_list.return_value = mock_response
            
            response = self.sagemaker_client.list_training_jobs(
                name_contains='job',
                status_equals='Completed',
                max_results=10
            )
            
            expected_params = {
                'MaxResults': 10,
                'SortBy': 'CreationTime',
                'SortOrder': 'Descending',
                'NameContains': 'job',
                'StatusEquals': 'Completed'
            }
            
            mock_list.assert_called_once_with(**expected_params)
            assert response == mock_response


class TestAWSAuthenticator:
    
    def setup_method(self):
        self.authenticator = AWSAuthenticator()
    
    @patch('boto3.Session')
    def test_get_aws_session_with_explicit_credentials(self, mock_session):
        inputs = {
            'aws-access-key-id': 'AKIAIOSFODNN7EXAMPLE',
            'aws-secret-access-key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            'aws-session-token': 'session-token',
            'aws-region': 'us-west-2'
        }
        
        session = self.authenticator.get_aws_session(inputs)
        
        mock_session.assert_called_once_with(
            aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
            aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            aws_session_token='session-token',
            region_name='us-west-2'
        )
    
    @patch('boto3.Session')
    def test_get_aws_session_default_credentials(self, mock_session):
        inputs = {
            'aws-region': 'eu-west-1'
        }
        
        session = self.authenticator.get_aws_session(inputs)
        
        mock_session.assert_called_once_with(region_name='eu-west-1')
    
    @patch.dict('os.environ', {'AWS_WEB_IDENTITY_TOKEN_FILE': '/tmp/token'})
    @patch('builtins.open', create=True)
    @patch('boto3.client')
    @patch('boto3.Session')
    def test_assume_role_with_oidc(self, mock_session, mock_client, mock_open):
        # Mock token file
        mock_open.return_value.__enter__.return_value.read.return_value.strip.return_value = 'mock-token'
        
        # Mock STS response
        mock_sts_client = Mock()
        mock_client.return_value = mock_sts_client
        mock_sts_client.assume_role_with_web_identity.return_value = {
            'Credentials': {
                'AccessKeyId': 'assumed-access-key',
                'SecretAccessKey': 'assumed-secret-key',
                'SessionToken': 'assumed-session-token'
            }
        }
        
        inputs = {
            'role-to-assume': 'arn:aws:iam::123456789012:role/TestRole',
            'aws-region': 'us-east-1'
        }
        
        session = self.authenticator.get_aws_session(inputs)
        
        # Verify assume_role_with_web_identity was called
        mock_sts_client.assume_role_with_web_identity.assert_called_once()
        
        # Verify session was created with assumed credentials
        mock_session.assert_called_with(
            aws_access_key_id='assumed-access-key',
            aws_secret_access_key='assumed-secret-key',
            aws_session_token='assumed-session-token',
            region_name='us-east-1'
        )
    
    @mock_sts
    def test_validate_credentials_success(self):
        session = boto3.Session(region_name='us-east-1')
        
        result = self.authenticator.validate_credentials(session)
        
        assert result is True
    
    def test_validate_credentials_failure(self):
        # Create session with invalid credentials
        session = boto3.Session(
            aws_access_key_id='invalid',
            aws_secret_access_key='invalid',
            region_name='us-east-1'
        )
        
        result = self.authenticator.validate_credentials(session)
        
        assert result is False