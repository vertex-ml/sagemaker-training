import json
import re
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class InputValidator:
    def __init__(self):
        self.logger = logger.bind(component="validator")
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        
        self.logger.info("Starting input validation")
        
        # Required fields validation
        required_fields = [
            'job-name',
            'algorithm-specification',
            'role-arn',
            'input-data-config',
            'output-data-config'
        ]
        
        for field in required_fields:
            if not inputs.get(field):
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Job name validation
        if inputs.get('job-name'):
            job_name_errors = self._validate_job_name(inputs['job-name'])
            errors.extend(job_name_errors)
        
        # Role ARN validation
        if inputs.get('role-arn'):
            role_arn_errors = self._validate_role_arn(inputs['role-arn'])
            errors.extend(role_arn_errors)
        
        # Instance type validation
        if inputs.get('instance-type'):
            instance_type_errors = self._validate_instance_type(inputs['instance-type'])
            errors.extend(instance_type_errors)
        
        # Numeric field validation
        numeric_fields = {
            'instance-count': (1, 100),
            'volume-size': (1, 16384),
            'max-runtime': (1, 432000),  # 5 days max
            'check-interval': (10, 3600)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if inputs.get(field):
                numeric_errors = self._validate_numeric_field(
                    field, inputs[field], min_val, max_val
                )
                errors.extend(numeric_errors)
        
        # JSON field validation
        json_fields = [
            'input-data-config',
            'output-data-config',
            'hyperparameters',
            'environment',
            'vpc-config',
            'tags'
        ]
        
        for field in json_fields:
            if inputs.get(field):
                json_errors = self._validate_json_field(field, inputs[field])
                errors.extend(json_errors)
        
        # Specific JSON structure validation
        if inputs.get('input-data-config'):
            input_config_errors = self._validate_input_data_config(inputs['input-data-config'])
            errors.extend(input_config_errors)
        
        if inputs.get('output-data-config'):
            output_config_errors = self._validate_output_data_config(inputs['output-data-config'])
            errors.extend(output_config_errors)
        
        if inputs.get('vpc-config'):
            vpc_config_errors = self._validate_vpc_config(inputs['vpc-config'])
            errors.extend(vpc_config_errors)
        
        # AWS region validation
        if inputs.get('aws-region'):
            region_warnings = self._validate_aws_region(inputs['aws-region'])
            warnings.extend(region_warnings)
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
        if result.is_valid:
            self.logger.info("Input validation passed", warnings=len(warnings))
        else:
            self.logger.error("Input validation failed", errors=len(errors))
        
        return result
    
    def _validate_job_name(self, job_name: str) -> List[str]:
        errors = []
        
        # SageMaker job name requirements
        if not re.match(r'^[a-zA-Z0-9\-]{1,63}$', job_name):
            errors.append(
                "Job name must be 1-63 characters long and contain only "
                "alphanumeric characters and hyphens"
            )
        
        if job_name.startswith('-') or job_name.endswith('-'):
            errors.append("Job name cannot start or end with a hyphen")
        
        return errors
    
    def _validate_role_arn(self, role_arn: str) -> List[str]:
        errors = []
        
        arn_pattern = r'^arn:aws(-[^:]*)?:iam::[0-9]{12}:role/.+$'
        if not re.match(arn_pattern, role_arn):
            errors.append("Role ARN format is invalid. Expected format: arn:aws:iam::account:role/role-name")
        
        return errors
    
    def _validate_instance_type(self, instance_type: str) -> List[str]:
        errors = []
        
        # Common SageMaker instance types pattern
        instance_pattern = r'^ml\.[a-z0-9]+\.(nano|micro|small|medium|large|xlarge|[0-9]+xlarge)$'
        if not re.match(instance_pattern, instance_type):
            errors.append(f"Instance type '{instance_type}' does not match expected SageMaker format")
        
        return errors
    
    def _validate_numeric_field(self, field: str, value: str, min_val: int, max_val: int) -> List[str]:
        errors = []
        
        try:
            num_value = int(value)
            if num_value < min_val or num_value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}")
        except ValueError:
            errors.append(f"{field} must be a valid integer")
        
        return errors
    
    def _validate_json_field(self, field: str, value: str) -> List[str]:
        errors = []
        
        try:
            json.loads(value)
        except json.JSONDecodeError as e:
            errors.append(f"{field} contains invalid JSON: {str(e)}")
        
        return errors
    
    def _validate_input_data_config(self, config_str: str) -> List[str]:
        errors = []
        
        try:
            config = json.loads(config_str)
            
            if not isinstance(config, list):
                errors.append("input-data-config must be a JSON array")
                return errors
            
            for i, channel in enumerate(config):
                if not isinstance(channel, dict):
                    errors.append(f"input-data-config[{i}] must be an object")
                    continue
                
                # Required fields for input channels
                required_channel_fields = ['ChannelName', 'DataSource']
                for field in required_channel_fields:
                    if field not in channel:
                        errors.append(f"input-data-config[{i}] missing required field: {field}")
                
                # Validate S3 data source
                if 'DataSource' in channel and 'S3DataSource' in channel['DataSource']:
                    s3_source = channel['DataSource']['S3DataSource']
                    if 'S3Uri' not in s3_source:
                        errors.append(f"input-data-config[{i}] S3DataSource missing S3Uri")
        
        except json.JSONDecodeError:
            pass  # Already handled in _validate_json_field
        
        return errors
    
    def _validate_output_data_config(self, config_str: str) -> List[str]:
        errors = []
        
        try:
            config = json.loads(config_str)
            
            if not isinstance(config, dict):
                errors.append("output-data-config must be a JSON object")
                return errors
            
            if 'S3OutputPath' not in config:
                errors.append("output-data-config missing required field: S3OutputPath")
            
            # Validate S3 URI format
            if 'S3OutputPath' in config:
                s3_uri = config['S3OutputPath']
                if not s3_uri.startswith('s3://'):
                    errors.append("S3OutputPath must be a valid S3 URI starting with s3://")
        
        except json.JSONDecodeError:
            pass  # Already handled in _validate_json_field
        
        return errors
    
    def _validate_vpc_config(self, config_str: str) -> List[str]:
        errors = []
        
        try:
            config = json.loads(config_str)
            
            if not isinstance(config, dict):
                errors.append("vpc-config must be a JSON object")
                return errors
            
            required_vpc_fields = ['SecurityGroupIds', 'Subnets']
            for field in required_vpc_fields:
                if field not in config:
                    errors.append(f"vpc-config missing required field: {field}")
                elif not isinstance(config[field], list):
                    errors.append(f"vpc-config.{field} must be an array")
        
        except json.JSONDecodeError:
            pass  # Already handled in _validate_json_field
        
        return errors
    
    def _validate_aws_region(self, region: str) -> List[str]:
        warnings = []
        
        # List of common AWS regions (not exhaustive)
        common_regions = [
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1',
            'ap-south-1', 'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1',
            'ap-northeast-2', 'sa-east-1', 'ca-central-1'
        ]
        
        if region not in common_regions:
            warnings.append(f"Region '{region}' is not in the list of common regions. Please verify it supports SageMaker.")
        
        return warnings