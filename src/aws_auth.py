import os
import boto3
from boto3.session import Session
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class AWSAuthenticator:
    def __init__(self):
        self.logger = logger.bind(component="aws_auth")
    
    def get_aws_session(self, inputs: Dict[str, Any]) -> Session:
        self.logger.info("Initializing AWS session")
        
        # Get AWS region
        region = inputs.get('aws-region', 'us-east-1')
        
        # Check for explicit credentials
        access_key = inputs.get('aws-access-key-id')
        secret_key = inputs.get('aws-secret-access-key')
        session_token = inputs.get('aws-session-token')
        
        if access_key and secret_key:
            self.logger.info("Using explicit AWS credentials")
            return Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
                region_name=region
            )
        
        # Check for role to assume (OIDC or existing credentials)
        role_arn = inputs.get('role-to-assume')
        if role_arn:
            self.logger.info(f"Assuming role: {role_arn}")
            return self._assume_role(role_arn, region)
        
        # Use default credential chain (environment, instance profile, etc.)
        self.logger.info("Using default AWS credential chain")
        return Session(region_name=region)
    
    def _assume_role(self, role_arn: str, region: str) -> Session:
        # Create STS client with current credentials
        sts_client = boto3.client('sts', region_name=region)
        
        # Check if we're using OIDC (GitHub Actions)
        web_identity_token_file = os.environ.get('AWS_WEB_IDENTITY_TOKEN_FILE')
        
        if web_identity_token_file and os.path.exists(web_identity_token_file):
            self.logger.info("Using OIDC web identity token for role assumption")
            
            with open(web_identity_token_file, 'r') as token_file:
                token = token_file.read().strip()
            
            response = sts_client.assume_role_with_web_identity(
                RoleArn=role_arn,
                RoleSessionName=os.environ.get('AWS_ROLE_SESSION_NAME', 'GitHubActions'),
                WebIdentityToken=token
            )
        else:
            self.logger.info("Using assume role with current credentials")
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=f'SageMakerTrainingAction-{os.environ.get("GITHUB_RUN_ID", "local")}'
            )
        
        credentials = response['Credentials']
        
        return Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=region
        )
    
    def validate_credentials(self, session: Session) -> bool:
        try:
            sts_client = session.client('sts')
            response = sts_client.get_caller_identity()
            
            self.logger.info(
                "AWS credentials validated",
                account_id=response.get('Account'),
                user_arn=response.get('Arn')
            )
            return True
            
        except Exception as e:
            self.logger.error(f"AWS credential validation failed: {str(e)}")
            return False