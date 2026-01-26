#!/bin/bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <repo_name> <tag> <source_account> <target_account> <region>"
    exit 1
fi

REPO_NAME=$1
TAG=$2
SOURCE_ACCOUNT=$3
TARGET_ACCOUNT=$4
REGION=$5

# Get the script directory path regardless of where the script is called from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine the base directory by going up from the script location
BASE_DIR="$( cd "${SCRIPT_DIR}/../../.." && pwd )"

# Check if all required directories exist, this is specific for TabArena
REQUIRED_DIRS=("autogluon" "tabarena")

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "${BASE_DIR}/${dir}" ]; then
        echo "Error: Required directory '${dir}' does not exist in ${BASE_DIR}"
        exit 1
    fi
done

echo "All required directories exist. Proceeding with Docker build."
echo "Using base directory: ${BASE_DIR}"
cp "${SCRIPT_DIR}/.dockerignore" "${BASE_DIR}/.dockerignore"

# Login to AWS ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${SOURCE_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${TARGET_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Build, tag and push the Docker image
docker build --no-cache -t ${REPO_NAME} -f "${SCRIPT_DIR}/Dockerfile_SM" ${BASE_DIR}

docker tag ${REPO_NAME}:latest ${TARGET_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}

docker push ${TARGET_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}