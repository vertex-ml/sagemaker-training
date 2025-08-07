import pytest
import json
from src.validators import InputValidator, ValidationResult


class TestInputValidator:
    
    def setup_method(self):
        self.validator = InputValidator()
        
        # Valid base inputs for testing
        self.valid_inputs = {
            'job-name': 'test-job-123',
            'algorithm-specification': '123456789012.dkr.ecr.us-east-1.amazonaws.com/my-algorithm:latest',
            'role-arn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
            'input-data-config': json.dumps([{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://my-bucket/training-data/'
                    }
                }
            }]),
            'output-data-config': json.dumps({
                'S3OutputPath': 's3://my-bucket/output/'
            })
        }
    
    def test_valid_inputs(self):
        result = self.validator.validate_inputs(self.valid_inputs)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_missing_required_fields(self):
        inputs = {}
        result = self.validator.validate_inputs(inputs)
        
        assert result.is_valid is False
        assert len(result.errors) == 5  # All required fields missing
        
        required_errors = [error for error in result.errors if 'Required field' in error]
        assert len(required_errors) == 5
    
    def test_invalid_job_name(self):
        test_cases = [
            ('', 'empty name'),
            ('job_with_underscores', 'underscores not allowed'),
            ('job-name-that-is-way-too-long-for-sagemaker-requirements-limit', 'too long'),
            ('-starting-with-hyphen', 'starts with hyphen'),
            ('ending-with-hyphen-', 'ends with hyphen'),
            ('job with spaces', 'contains spaces')
        ]
        
        for invalid_name, description in test_cases:
            inputs = self.valid_inputs.copy()
            inputs['job-name'] = invalid_name
            
            result = self.validator.validate_inputs(inputs)
            
            assert result.is_valid is False, f"Should fail for {description}: {invalid_name}"
            job_name_errors = [error for error in result.errors if 'Job name' in error]
            assert len(job_name_errors) > 0, f"Should have job name error for {description}"
    
    def test_valid_job_names(self):
        valid_names = [
            'simple-job',
            'job123',
            'a',
            'Job-With-Mixed-Case',
            'job-' + '1' * 59  # 63 characters total
        ]
        
        for valid_name in valid_names:
            inputs = self.valid_inputs.copy()
            inputs['job-name'] = valid_name
            
            result = self.validator.validate_inputs(inputs)
            
            # Check that job name validation passes
            job_name_errors = [error for error in result.errors if 'Job name' in error]
            assert len(job_name_errors) == 0, f"Should not have job name errors for: {valid_name}"
    
    def test_invalid_role_arn(self):
        invalid_arns = [
            'not-an-arn',
            'arn:aws:iam::123456789012:user/username',  # user, not role
            'arn:aws:iam::invalid:role/rolename',  # invalid account
            'arn:aws:s3:::bucket-name',  # wrong service
        ]
        
        for invalid_arn in invalid_arns:
            inputs = self.valid_inputs.copy()
            inputs['role-arn'] = invalid_arn
            
            result = self.validator.validate_inputs(inputs)
            
            assert result.is_valid is False
            role_arn_errors = [error for error in result.errors if 'Role ARN' in error]
            assert len(role_arn_errors) > 0
    
    def test_valid_role_arns(self):
        valid_arns = [
            'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
            'arn:aws:iam::999999999999:role/MyRole',
            'arn:aws-cn:iam::123456789012:role/ChinaRole',  # China partition
            'arn:aws-us-gov:iam::123456789012:role/GovRole'  # GovCloud partition
        ]
        
        for valid_arn in valid_arns:
            inputs = self.valid_inputs.copy()
            inputs['role-arn'] = valid_arn
            
            result = self.validator.validate_inputs(inputs)
            
            role_arn_errors = [error for error in result.errors if 'Role ARN' in error]
            assert len(role_arn_errors) == 0
    
    def test_invalid_instance_type(self):
        invalid_types = [
            't2.micro',  # not ml instance
            'ml.invalid.size',  # invalid format
            'm5.large',  # missing ml prefix
            'ml.m5.invalid-size'  # invalid size
        ]
        
        for invalid_type in invalid_types:
            inputs = self.valid_inputs.copy()
            inputs['instance-type'] = invalid_type
            
            result = self.validator.validate_inputs(inputs)
            
            instance_type_errors = [error for error in result.errors if 'Instance type' in error]
            assert len(instance_type_errors) > 0
    
    def test_valid_instance_types(self):
        valid_types = [
            'ml.m5.large',
            'ml.c5.xlarge',
            'ml.p3.2xlarge',
            'ml.g4dn.12xlarge',
            'ml.r5.24xlarge'
        ]
        
        for valid_type in valid_types:
            inputs = self.valid_inputs.copy()
            inputs['instance-type'] = valid_type
            
            result = self.validator.validate_inputs(inputs)
            
            instance_type_errors = [error for error in result.errors if 'Instance type' in error]
            assert len(instance_type_errors) == 0
    
    def test_numeric_field_validation(self):
        test_cases = [
            ('instance-count', '0', 'below minimum'),
            ('instance-count', '101', 'above maximum'),
            ('instance-count', 'not-a-number', 'not numeric'),
            ('volume-size', '0', 'below minimum'),
            ('volume-size', '20000', 'above maximum'),
            ('max-runtime', '0', 'below minimum'),
            ('max-runtime', '500000', 'above maximum')
        ]
        
        for field, value, description in test_cases:
            inputs = self.valid_inputs.copy()
            inputs[field] = value
            
            result = self.validator.validate_inputs(inputs)
            
            field_errors = [error for error in result.errors if field in error]
            assert len(field_errors) > 0, f"Should have error for {field} {description}: {value}"
    
    def test_invalid_json_fields(self):
        json_fields = [
            'input-data-config',
            'output-data-config',
            'hyperparameters',
            'environment',
            'vpc-config',
            'tags'
        ]
        
        invalid_json = '{"invalid": json, "missing": quotes}'
        
        for field in json_fields:
            inputs = self.valid_inputs.copy()
            inputs[field] = invalid_json
            
            result = self.validator.validate_inputs(inputs)
            
            json_errors = [error for error in result.errors if field in error and 'JSON' in error]
            assert len(json_errors) > 0, f"Should have JSON error for field: {field}"
    
    def test_input_data_config_validation(self):
        # Test invalid structure - not an array
        inputs = self.valid_inputs.copy()
        inputs['input-data-config'] = json.dumps({"not": "array"})
        
        result = self.validator.validate_inputs(inputs)
        assert result.is_valid is False
        array_errors = [error for error in result.errors if 'must be a JSON array' in error]
        assert len(array_errors) > 0
        
        # Test missing required fields
        inputs['input-data-config'] = json.dumps([{"missing": "required_fields"}])
        result = self.validator.validate_inputs(inputs)
        
        required_field_errors = [error for error in result.errors if 'missing required field' in error]
        assert len(required_field_errors) > 0
    
    def test_output_data_config_validation(self):
        # Test invalid structure - not an object
        inputs = self.valid_inputs.copy()
        inputs['output-data-config'] = json.dumps(["not", "object"])
        
        result = self.validator.validate_inputs(inputs)
        assert result.is_valid is False
        
        # Test missing S3OutputPath
        inputs['output-data-config'] = json.dumps({"missing": "s3_output_path"})
        result = self.validator.validate_inputs(inputs)
        
        s3_errors = [error for error in result.errors if 'S3OutputPath' in error]
        assert len(s3_errors) > 0
        
        # Test invalid S3 URI
        inputs['output-data-config'] = json.dumps({"S3OutputPath": "invalid-uri"})
        result = self.validator.validate_inputs(inputs)
        
        uri_errors = [error for error in result.errors if 'valid S3 URI' in error]
        assert len(uri_errors) > 0
    
    def test_vpc_config_validation(self):
        inputs = self.valid_inputs.copy()
        
        # Test invalid structure
        inputs['vpc-config'] = json.dumps(["not", "object"])
        result = self.validator.validate_inputs(inputs)
        
        structure_errors = [error for error in result.errors if 'must be a JSON object' in error]
        assert len(structure_errors) > 0
        
        # Test missing required fields
        inputs['vpc-config'] = json.dumps({"missing": "required_fields"})
        result = self.validator.validate_inputs(inputs)
        
        missing_errors = [error for error in result.errors if 'vpc-config missing required field' in error]
        assert len(missing_errors) > 0
        
        # Test non-array fields
        inputs['vpc-config'] = json.dumps({
            "SecurityGroupIds": "not-array",
            "Subnets": "not-array"
        })
        result = self.validator.validate_inputs(inputs)
        
        array_errors = [error for error in result.errors if 'must be an array' in error]
        assert len(array_errors) > 0
    
    def test_aws_region_warnings(self):
        inputs = self.valid_inputs.copy()
        
        # Test unknown region
        inputs['aws-region'] = 'unknown-region-1'
        result = self.validator.validate_inputs(inputs)
        
        # Should still be valid but have warnings
        assert result.is_valid is True
        region_warnings = [warning for warning in result.warnings if 'not in the list of common regions' in warning]
        assert len(region_warnings) > 0
        
        # Test known region
        inputs['aws-region'] = 'us-east-1'
        result = self.validator.validate_inputs(inputs)
        
        region_warnings = [warning for warning in result.warnings if 'not in the list of common regions' in warning]
        assert len(region_warnings) == 0