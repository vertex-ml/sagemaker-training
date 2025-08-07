#!/bin/bash

set -e

# Set up AWS credentials if provided
if [[ -n "$INPUT_AWS_ACCESS_KEY_ID" && -n "$INPUT_AWS_SECRET_ACCESS_KEY" ]]; then
    export AWS_ACCESS_KEY_ID="$INPUT_AWS_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$INPUT_AWS_SECRET_ACCESS_KEY"
    
    if [[ -n "$INPUT_AWS_SESSION_TOKEN" ]]; then
        export AWS_SESSION_TOKEN="$INPUT_AWS_SESSION_TOKEN"
    fi
fi

# Set AWS region
if [[ -n "$INPUT_AWS_REGION" ]]; then
    export AWS_DEFAULT_REGION="$INPUT_AWS_REGION"
    export AWS_REGION="$INPUT_AWS_REGION"
fi

# Handle role assumption if specified
if [[ -n "$INPUT_ROLE_TO_ASSUME" ]]; then
    echo "Assuming role: $INPUT_ROLE_TO_ASSUME"
    
    # Use GitHub OIDC token to assume role
    if [[ -n "$ACTIONS_ID_TOKEN_REQUEST_TOKEN" && -n "$ACTIONS_ID_TOKEN_REQUEST_URL" ]]; then
        export AWS_WEB_IDENTITY_TOKEN_FILE="/tmp/web-identity-token"
        curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
             "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=sts.amazonaws.com" | \
             jq -r '.value' > "$AWS_WEB_IDENTITY_TOKEN_FILE"
        
        export AWS_ROLE_ARN="$INPUT_ROLE_TO_ASSUME"
        export AWS_ROLE_SESSION_NAME="GitHubActions-$(date +%s)"
    fi
fi

# Validate AWS credentials
echo "Validating AWS credentials..."
aws sts get-caller-identity > /dev/null || {
    echo "ERROR: AWS credentials validation failed"
    exit 1
}

echo "AWS credentials validated successfully"

# Run the main Python script
echo "Starting SageMaker Training Action..."
cd /app
python -m src.main