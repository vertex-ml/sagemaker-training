import os
import sys
import structlog
from typing import Any, Optional


def setup_logging() -> structlog.stdlib.BoundLogger:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger("sagemaker_training_action")
    
    # Set log level from environment
    log_level = os.environ.get('ACTIONS_RUNNER_DEBUG', '').lower()
    if log_level == 'true':
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    return logger


def get_github_output() -> Optional[str]:
    return os.environ.get('GITHUB_OUTPUT')


def set_github_output(name: str, value: Any) -> None:
    github_output = get_github_output()
    
    if github_output:
        # Write to GitHub Actions output file
        with open(github_output, 'a', encoding='utf-8') as f:
            f.write(f"{name}={value}\n")
    else:
        # Fallback to environment variable for local testing
        os.environ[f'OUTPUT_{name.upper().replace("-", "_")}'] = str(value)
    
    # Also print to stdout for visibility
    print(f"::set-output name={name}::{value}")


def get_github_input(name: str, default: Optional[str] = None) -> Optional[str]:
    env_name = f'INPUT_{name.upper().replace("-", "_")}'
    return os.environ.get(env_name, default)


def mask_sensitive_value(value: str) -> None:
    print(f"::add-mask::{value}")


def set_github_env_var(name: str, value: str) -> None:
    github_env = os.environ.get('GITHUB_ENV')
    
    if github_env:
        with open(github_env, 'a', encoding='utf-8') as f:
            f.write(f"{name}={value}\n")
    else:
        os.environ[name] = value


def github_warning(message: str, file: Optional[str] = None, line: Optional[int] = None) -> None:
    warning_cmd = "::warning"
    
    if file:
        warning_cmd += f" file={file}"
    if line:
        warning_cmd += f",line={line}"
    
    print(f"{warning_cmd}::{message}")


def github_error(message: str, file: Optional[str] = None, line: Optional[int] = None) -> None:
    error_cmd = "::error"
    
    if file:
        error_cmd += f" file={file}"
    if line:
        error_cmd += f",line={line}"
    
    print(f"{error_cmd}::{message}")


def github_notice(message: str, file: Optional[str] = None, line: Optional[int] = None) -> None:
    notice_cmd = "::notice"
    
    if file:
        notice_cmd += f" file={file}"
    if line:
        notice_cmd += f",line={line}"
    
    print(f"{notice_cmd}::{message}")


def create_github_summary(content: str) -> None:
    github_step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    
    if github_step_summary:
        with open(github_step_summary, 'a', encoding='utf-8') as f:
            f.write(content)
    else:
        print("GitHub Step Summary:")
        print(content)


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def sanitize_job_name(name: str) -> str:
    import re
    
    # Remove invalid characters and replace with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9\-]', '-', name)
    
    # Remove consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    
    # Ensure maximum length
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip('-')
    
    # Ensure minimum length
    if len(sanitized) == 0:
        sanitized = 'sagemaker-job'
    
    return sanitized


def format_s3_uri(bucket: str, key: str) -> str:
    if not bucket.startswith('s3://'):
        bucket = f's3://{bucket}'
    
    if key.startswith('/'):
        key = key[1:]
    
    return f"{bucket.rstrip('/')}/{key}"


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    # Remove s3:// prefix
    path = s3_uri[5:]
    
    # Split bucket and key
    parts = path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket, key