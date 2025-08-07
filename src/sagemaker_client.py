import time
import json
from typing import Dict, Any, Optional
from boto3.session import Session
import structlog

logger = structlog.get_logger()


class SageMakerClient:
    def __init__(self, session: Session, region: str):
        self.session = session
        self.region = region
        self.sagemaker = session.client('sagemaker', region_name=region)
        self.logger = logger.bind(component="sagemaker_client")
    
    def create_training_job(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(
            "Creating SageMaker training job",
            job_name=training_config['TrainingJobName']
        )
        
        try:
            response = self.sagemaker.create_training_job(**training_config)
            
            self.logger.info(
                "Training job created successfully",
                job_name=training_config['TrainingJobName'],
                job_arn=response['TrainingJobArn']
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                f"Failed to create training job: {str(e)}",
                job_name=training_config['TrainingJobName']
            )
            raise
    
    def describe_training_job(self, job_name: str) -> Dict[str, Any]:
        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            return response
        except Exception as e:
            self.logger.error(f"Failed to describe training job {job_name}: {str(e)}")
            raise
    
    def wait_for_training_job_completion(
        self, 
        job_name: str, 
        check_interval: int = 60,
        max_wait_time: int = 86400
    ) -> str:
        self.logger.info(
            "Waiting for training job completion",
            job_name=job_name,
            check_interval=check_interval
        )
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                job_details = self.describe_training_job(job_name)
                status = job_details['TrainingJobStatus']
                
                self.logger.info(
                    "Training job status check",
                    job_name=job_name,
                    status=status,
                    elapsed_time=int(time.time() - start_time)
                )
                
                # Terminal states
                if status in ['Completed', 'Failed', 'Stopped']:
                    if status == 'Completed':
                        self.logger.info(
                            "Training job completed successfully",
                            job_name=job_name,
                            total_time=int(time.time() - start_time)
                        )
                    else:
                        failure_reason = job_details.get('FailureReason', 'Unknown')
                        self.logger.error(
                            f"Training job finished with status: {status}",
                            job_name=job_name,
                            failure_reason=failure_reason
                        )
                    
                    return status
                
                # Log secondary status if available
                secondary_status = job_details.get('SecondaryStatus')
                if secondary_status:
                    self.logger.debug(
                        "Training job secondary status",
                        job_name=job_name,
                        secondary_status=secondary_status
                    )
                
                # Wait before next check
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(
                    f"Error checking training job status: {str(e)}",
                    job_name=job_name
                )
                raise
        
        # Timeout reached
        self.logger.error(
            "Timeout waiting for training job completion",
            job_name=job_name,
            max_wait_time=max_wait_time
        )
        raise TimeoutError(f"Training job {job_name} did not complete within {max_wait_time} seconds")
    
    def stop_training_job(self, job_name: str) -> None:
        self.logger.info(f"Stopping training job: {job_name}")
        
        try:
            self.sagemaker.stop_training_job(TrainingJobName=job_name)
            self.logger.info(f"Stop request sent for training job: {job_name}")
        except Exception as e:
            self.logger.error(f"Failed to stop training job {job_name}: {str(e)}")
            raise
    
    def list_training_jobs(
        self, 
        name_contains: Optional[str] = None,
        status_equals: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        params = {
            'MaxResults': max_results,
            'SortBy': 'CreationTime',
            'SortOrder': 'Descending'
        }
        
        if name_contains:
            params['NameContains'] = name_contains
        
        if status_equals:
            params['StatusEquals'] = status_equals
        
        try:
            response = self.sagemaker.list_training_jobs(**params)
            return response
        except Exception as e:
            self.logger.error(f"Failed to list training jobs: {str(e)}")
            raise
    
    def get_training_job_logs(self, job_name: str) -> Optional[str]:
        try:
            # Get CloudWatch logs for the training job
            logs_client = self.session.client('logs', region_name=self.region)
            
            log_group_name = f'/aws/sagemaker/TrainingJobs'
            log_stream_name = f'{job_name}/algo-1-1234567890'
            
            response = logs_client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                limit=100
            )
            
            logs = []
            for event in response.get('events', []):
                logs.append(event['message'])
            
            return '\n'.join(logs)
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve logs for job {job_name}: {str(e)}")
            return None