import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from src.utils import (
    sanitize_job_name,
    format_duration,
    format_s3_uri,
    parse_s3_uri,
    set_github_output,
    get_github_input,
    create_github_summary
)


class TestUtilityFunctions:
    
    def test_sanitize_job_name(self):
        test_cases = [
            # (input, expected_output, description)
            ('valid-job-name', 'valid-job-name', 'already valid'),
            ('job_with_underscores', 'job-with-underscores', 'replace underscores'),
            ('job with spaces', 'job-with-spaces', 'replace spaces'),
            ('job!!!with###special$$$chars', 'job-with-special-chars', 'replace special chars'),
            ('---multiple---hyphens---', 'multiple-hyphens', 'consolidate hyphens'),
            ('-starting-and-ending-', 'starting-and-ending', 'remove leading/trailing hyphens'),
            ('', 'sagemaker-job', 'empty string default'),
            ('a' * 100, 'a' * 63, 'truncate long names'),
            ('job@domain.com', 'job-domain-com', 'replace @ and .'),
            ('Job-With-CAPS', 'Job-With-CAPS', 'preserve case'),
            ('123-numeric-start', '123-numeric-start', 'numeric start ok')
        ]
        
        for input_name, expected, description in test_cases:
            result = sanitize_job_name(input_name)
            assert result == expected, f"Failed for {description}: got '{result}', expected '{expected}'"
            
            # Ensure result meets SageMaker requirements
            assert len(result) <= 63, f"Result too long: {result}"
            assert len(result) > 0, f"Result empty: {result}"
            assert not result.startswith('-'), f"Result starts with hyphen: {result}"
            assert not result.endswith('-'), f"Result ends with hyphen: {result}"
    
    def test_format_duration(self):
        test_cases = [
            (30, '30s'),
            (90, '1m 30s'),
            (3661, '1h 1m 1s'),
            (7200, '2h 0m 0s'),
            (3600, '1h 0m 0s'),
            (60, '1m 0s'),
            (0, '0s')
        ]
        
        for seconds, expected in test_cases:
            result = format_duration(seconds)
            assert result == expected, f"Duration formatting failed for {seconds}s: got '{result}', expected '{expected}'"
    
    def test_format_s3_uri(self):
        test_cases = [
            ('my-bucket', 'path/to/file.txt', 's3://my-bucket/path/to/file.txt'),
            ('s3://my-bucket', 'path/to/file.txt', 's3://my-bucket/path/to/file.txt'),
            ('my-bucket/', '/path/to/file.txt', 's3://my-bucket/path/to/file.txt'),
            ('s3://my-bucket/', '/path/to/file.txt', 's3://my-bucket/path/to/file.txt'),
            ('my-bucket', '', 's3://my-bucket/'),
            ('my-bucket', '/', 's3://my-bucket/')
        ]
        
        for bucket, key, expected in test_cases:
            result = format_s3_uri(bucket, key)
            assert result == expected, f"S3 URI formatting failed: got '{result}', expected '{expected}'"
    
    def test_parse_s3_uri(self):
        test_cases = [
            ('s3://my-bucket/path/to/file.txt', ('my-bucket', 'path/to/file.txt')),
            ('s3://my-bucket/', ('my-bucket', '')),
            ('s3://my-bucket', ('my-bucket', '')),
            ('s3://bucket-with-hyphens/deep/nested/path/file.json', ('bucket-with-hyphens', 'deep/nested/path/file.json'))
        ]
        
        for s3_uri, expected in test_cases:
            result = parse_s3_uri(s3_uri)
            assert result == expected, f"S3 URI parsing failed for '{s3_uri}': got {result}, expected {expected}"
        
        # Test invalid URIs
        invalid_uris = [
            'http://example.com',
            'not-a-uri',
            's3://',
            'bucket/path'
        ]
        
        for invalid_uri in invalid_uris:
            with pytest.raises(ValueError, match="Invalid S3 URI"):
                parse_s3_uri(invalid_uri)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_github_input(self):
        # Test with environment variable set
        with patch.dict(os.environ, {'INPUT_JOB_NAME': 'test-job'}):
            result = get_github_input('job-name')
            assert result == 'test-job'
        
        # Test with default value
        result = get_github_input('non-existent', 'default-value')
        assert result == 'default-value'
        
        # Test with no value and no default
        result = get_github_input('non-existent')
        assert result is None
        
        # Test with hyphenated input name
        with patch.dict(os.environ, {'INPUT_AWS_REGION': 'us-west-2'}):
            result = get_github_input('aws-region')
            assert result == 'us-west-2'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch.dict(os.environ, {'GITHUB_OUTPUT': '/tmp/github_output'})
    def test_set_github_output_with_file(self, mock_file):
        set_github_output('test-key', 'test-value')
        
        # Verify file was opened and written to
        mock_file.assert_called_once_with('/tmp/github_output', 'a', encoding='utf-8')
        mock_file().write.assert_called_once_with('test-key=test-value\n')
    
    @patch.dict(os.environ, {}, clear=True)
    def test_set_github_output_without_file(self):
        set_github_output('test-key', 'test-value')
        
        # Should set environment variable as fallback
        assert os.environ.get('OUTPUT_TEST_KEY') == 'test-value'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': '/tmp/step_summary'})
    def test_create_github_summary_with_file(self, mock_file):
        content = "# Test Summary\nThis is a test summary."
        create_github_summary(content)
        
        mock_file.assert_called_once_with('/tmp/step_summary', 'a', encoding='utf-8')
        mock_file().write.assert_called_once_with(content)
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('builtins.print')
    def test_create_github_summary_without_file(self, mock_print):
        content = "# Test Summary\nThis is a test summary."
        create_github_summary(content)
        
        # Should print to console as fallback
        mock_print.assert_any_call("GitHub Step Summary:")
        mock_print.assert_any_call(content)