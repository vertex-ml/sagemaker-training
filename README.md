# SageMaker Training Action

[![GitHub Marketplace](https://img.shields.io/badge/GitHub%20Marketplace-SageMaker%20Training-blue?logo=github)](https://github.com/marketplace/actions/sagemaker-training-action)
[![Tests](https://github.com/your-org/sagemaker-training-action/workflows/Tests/badge.svg)](https://github.com/your-org/sagemaker-training-action/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive GitHub Action for setting up and triggering Amazon SageMaker training jobs with full MLOps integration support.

## 🚀 Features

- **Easy Setup**: Simple configuration with sensible defaults
- **Flexible Authentication**: Support for AWS credentials, IAM roles, and OIDC
- **Comprehensive Validation**: Built-in input validation with clear error messages
- **Rich Monitoring**: Real-time job status tracking and logging
- **VPC Support**: Secure training in private networks
- **Custom Containers**: Support for both built-in algorithms and custom Docker images
- **Hyperparameter Management**: Easy configuration of training hyperparameters
- **Resource Tagging**: Automated resource tagging for cost management
- **Multi-Instance Training**: Support for distributed training across multiple instances

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Usage Examples](#usage-examples)
- [Advanced Configuration](#advanced-configuration)
- [Authentication](#authentication)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🏃‍♂️ Quick Start

### Basic Usage

```yaml
name: Train ML Model

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: us-east-1
      
      - name: Train SageMaker Model
        uses: your-org/sagemaker-training-action@v1
        with:
          job-name: my-training-job-${{ github.run_number }}
          algorithm-specification: 382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest
          role-arn: arn:aws:iam::123456789012:role/SageMakerExecutionRole
          instance-type: ml.m5.large
          input-data-config: |
            [{
              \"ChannelName\": \"training\",
              \"DataSource\": {
                \"S3DataSource\": {
                  \"S3DataType\": \"S3Prefix\",
                  \"S3Uri\": \"s3://my-training-bucket/data/\"
                }
              }
            }]
          output-data-config: |
            {
              \"S3OutputPath\": \"s3://my-training-bucket/output/\"
            }
```

## 📥 Inputs

### Required Inputs

| Input | Description | Example |
|-------|-------------|---------|
| `job-name` | SageMaker training job name (1-63 chars, alphanumeric and hyphens only) | `my-training-job` |
| `algorithm-specification` | Training container image URI or built-in algorithm | `382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest` |
| `role-arn` | SageMaker execution role ARN | `arn:aws:iam::123456789012:role/SageMakerExecutionRole` |
| `input-data-config` | JSON array of input data channels | See [examples](#input-data-configuration) |
| `output-data-config` | JSON object for output configuration | `{\"S3OutputPath\": \"s3://bucket/output/\"}` |

### Optional Inputs

| Input | Description | Default | Example |
|-------|-------------|---------|---------|
| `aws-region` | AWS region for SageMaker | `us-east-1` | `us-west-2` |
| `instance-type` | ML compute instance type | `ml.m5.large` | `ml.c5.xlarge` |
| `instance-count` | Number of ML compute instances | `1` | `2` |
| `volume-size` | EBS volume size in GB | `30` | `100` |
| `max-runtime` | Maximum training time in seconds | `86400` | `172800` |
| `hyperparameters` | JSON object of hyperparameters | `{}` | `{\"max_depth\": \"5\"}` |
| `environment` | JSON object of environment variables | `{}` | `{\"PYTHONPATH\": \"/opt/ml/code\"}` |
| `vpc-config` | VPC configuration for secure training | | See [VPC example](#vpc-configuration) |
| `tags` | Resource tags as JSON object | `{}` | `{\"Project\": \"MLOps\"}` |
| `wait-for-completion` | Wait for job completion | `true` | `false` |
| `check-interval` | Status check interval in seconds | `60` | `120` |

### Authentication Inputs

| Input | Description | Required |
|-------|-------------|----------|
| `aws-access-key-id` | AWS Access Key ID | If not using OIDC |
| `aws-secret-access-key` | AWS Secret Access Key | If not using OIDC |
| `aws-session-token` | AWS Session Token | For temporary credentials |
| `role-to-assume` | IAM Role ARN to assume | For OIDC authentication |

## 📤 Outputs

| Output | Description | Example |
|--------|-------------|---------|
| `job-name` | The SageMaker training job name | `my-training-job-123` |
| `job-arn` | The training job ARN | `arn:aws:sagemaker:us-east-1:123456789012:training-job/my-job` |
| `job-status` | Final job status | `Completed` |
| `model-artifacts` | S3 URI of model artifacts | `s3://bucket/output/model.tar.gz` |
| `training-image` | Container image used for training | `382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest` |
| `training-job-definition` | Complete job configuration as JSON | Full job definition |

## 📚 Usage Examples

### Basic XGBoost Training

```yaml
- name: Train XGBoost Model
  uses: your-org/sagemaker-training-action@v1
  with:
    job-name: xgboost-model-${{ github.sha }}
    algorithm-specification: 382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest
    role-arn: ${{ secrets.SAGEMAKER_ROLE_ARN }}
    instance-type: ml.m5.xlarge
    input-data-config: |
      [{
        \"ChannelName\": \"training\",
        \"DataSource\": {
          \"S3DataSource\": {
            \"S3DataType\": \"S3Prefix\",
            \"S3Uri\": \"s3://my-bucket/training-data/\"
          }
        },
        \"ContentType\": \"text/csv\"
      }]
    output-data-config: |
      {
        \"S3OutputPath\": \"s3://my-bucket/models/\"
      }
    hyperparameters: |
      {
        \"max_depth\": \"6\",
        \"eta\": \"0.3\",
        \"num_round\": \"100\",
        \"subsample\": \"0.8\"
      }
```

### Custom Container Training

```yaml
- name: Train Custom Algorithm
  uses: your-org/sagemaker-training-action@v1
  with:
    job-name: custom-algorithm-${{ github.run_number }}
    algorithm-specification: ${{ secrets.ECR_REGISTRY }}/my-algorithm:${{ github.sha }}
    role-arn: ${{ secrets.SAGEMAKER_ROLE_ARN }}
    instance-type: ml.c5.2xlarge
    instance-count: 2
    volume-size: 100
    input-data-config: |
      [{
        \"ChannelName\": \"training\",
        \"DataSource\": {
          \"S3DataSource\": {
            \"S3DataType\": \"S3Prefix\",
            \"S3Uri\": \"s3://data-bucket/train/\"
          }
        }
      }, {
        \"ChannelName\": \"validation\",
        \"DataSource\": {
          \"S3DataSource\": {
            \"S3DataType\": \"S3Prefix\",
            \"S3Uri\": \"s3://data-bucket/validation/\"
          }
        }
      }]
    output-data-config: |
      {
        \"S3OutputPath\": \"s3://model-bucket/artifacts/\",
        \"KmsKeyId\": \"${{ secrets.KMS_KEY_ID }}\"
      }
    environment: |
      {
        \"SAGEMAKER_PROGRAM\": \"train.py\",
        \"SAGEMAKER_REQUIREMENTS\": \"requirements.txt\",
        \"MODEL_VERSION\": \"${{ github.sha }}\"
      }
    tags: |
      {
        \"Repository\": \"${{ github.repository }}\",
        \"CommitSHA\": \"${{ github.sha }}\",
        \"Environment\": \"production\"
      }
```

### Multi-Instance Distributed Training

```yaml
- name: Distributed Training
  uses: your-org/sagemaker-training-action@v1
  with:
    job-name: distributed-training-${{ github.run_number }}
    algorithm-specification: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py3
    role-arn: ${{ secrets.SAGEMAKER_ROLE_ARN }}
    instance-type: ml.p3.8xlarge
    instance-count: 4
    volume-size: 200
    max-runtime: 172800  # 48 hours
    input-data-config: |
      [{
        \"ChannelName\": \"training\",
        \"DataSource\": {
          \"S3DataSource\": {
            \"S3DataType\": \"S3Prefix\",
            \"S3Uri\": \"s3://large-dataset-bucket/training/\",
            \"S3DataDistributionType\": \"ShardedByS3Key\"
          }
        },
        \"InputMode\": \"FastFile\"
      }]
    output-data-config: |
      {
        \"S3OutputPath\": \"s3://model-artifacts-bucket/distributed-model/\"
      }
    hyperparameters: |
      {
        \"epochs\": \"50\",
        \"batch-size\": \"64\",
        \"learning-rate\": \"0.001\",
        \"backend\": \"nccl\"
      }
```

### VPC Configuration

```yaml
- name: Secure VPC Training
  uses: your-org/sagemaker-training-action@v1
  with:
    job-name: secure-training-${{ github.run_number }}
    algorithm-specification: 382416733822.dkr.ecr.us-east-1.amazonaws.com/scikit-learn:latest
    role-arn: ${{ secrets.SAGEMAKER_VPC_ROLE_ARN }}
    instance-type: ml.m5.large
    input-data-config: |
      [{
        \"ChannelName\": \"training\",
        \"DataSource\": {
          \"S3DataSource\": {
            \"S3DataType\": \"S3Prefix\",
            \"S3Uri\": \"s3://private-data-bucket/sensitive-data/\"
          }
        }
      }]
    output-data-config: |
      {
        \"S3OutputPath\": \"s3://private-models-bucket/secure-output/\"
      }
    vpc-config: |
      {
        \"SecurityGroupIds\": [\"${{ secrets.SAGEMAKER_SECURITY_GROUP }}\"],
        \"Subnets\": [\"${{ secrets.PRIVATE_SUBNET_1 }}\", \"${{ secrets.PRIVATE_SUBNET_2 }}\"]
      }
```

## 🔐 Authentication

### OIDC Authentication (Recommended)

Use GitHub's OIDC provider for secure, keyless authentication:

```yaml
permissions:
  id-token: write
  contents: read

steps:
  - name: Configure AWS credentials
    uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
      aws-region: us-east-1
  
  - name: Train Model
    uses: your-org/sagemaker-training-action@v1
    with:
      # ... other inputs
```

### IAM User Credentials

Store AWS credentials as repository secrets:

```yaml
- name: Train Model
  uses: your-org/sagemaker-training-action@v1
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    # ... other inputs
```

## 🔧 Advanced Configuration

### Input Data Configuration

The `input-data-config` parameter accepts a JSON array of data channels:

```json
[
  {
    \"ChannelName\": \"training\",
    \"DataSource\": {
      \"S3DataSource\": {
        \"S3DataType\": \"S3Prefix\",
        \"S3Uri\": \"s3://bucket/training/\",
        \"S3DataDistributionType\": \"FullyReplicated\"
      }
    },
    \"ContentType\": \"text/csv\",
    \"CompressionType\": \"Gzip\",
    \"InputMode\": \"File\"
  }
]
```

### Output Data Configuration

```json
{
  \"S3OutputPath\": \"s3://bucket/output/\",
  \"KmsKeyId\": \"arn:aws:kms:region:account:key/key-id\"
}
```

### Environment Variables

Pass custom environment variables to your training container:

```json
{
  \"PYTHONPATH\": \"/opt/ml/code\",
  \"CUDA_VISIBLE_DEVICES\": \"0,1\",
  \"OMP_NUM_THREADS\": \"4\"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Job Name Validation Errors**
   - Ensure job names are 1-63 characters
   - Use only alphanumeric characters and hyphens
   - Don't start or end with hyphens

2. **Permission Denied Errors**
   - Verify SageMaker execution role has necessary permissions
   - Check S3 bucket policies and permissions
   - Ensure ECR repository access for custom containers

3. **Instance Type Errors**
   - Use valid SageMaker instance types (ml.*)
   - Check instance availability in your region
   - Verify service limits for instance types

4. **Timeout Issues**
   - Increase `max-runtime` for long-running jobs
   - Optimize data loading and preprocessing
   - Consider using faster instance types

### Debug Mode

Enable debug logging by setting the repository variable `ACTIONS_RUNNER_DEBUG` to `true`.

### Getting Help

- Check [troubleshooting guide](docs/troubleshooting.md)
- Review [AWS SageMaker documentation](https://docs.aws.amazon.com/sagemaker/)
- Open an [issue](https://github.com/your-org/sagemaker-training-action/issues) for bugs or feature requests

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`
4. Build container: `docker build -t sagemaker-training-action .`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For available versions, see the [releases on this repository](https://github.com/your-org/sagemaker-training-action/releases).

## 📞 Support

- **Documentation**: [GitHub Pages](https://your-org.github.io/sagemaker-training-action/)
- **Issues**: [GitHub Issues](https://github.com/your-org/sagemaker-training-action/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/sagemaker-training-action/discussions)

---

**Made with ❤️ for the MLOps community**