#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load configuration from .env.cluster
ENV_FILE="$SCRIPT_DIR/.env.cluster"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Configuration file not found: $ENV_FILE"
    echo "Please create .env.cluster with CLUSTER_LOGIN and CLUSTER_BASE_DIR variables"
    exit 1
fi

# Source the environment file
source "$ENV_FILE"

# Validate required variables
if [ -z "$CLUSTER_LOGIN" ]; then
    echo "Error: CLUSTER_LOGIN not set in $ENV_FILE"
    exit 1
fi

if [ -z "$CLUSTER_BASE_DIR" ]; then
    echo "Error: CLUSTER_BASE_DIR not set in $ENV_FILE"
    exit 1
fi

# Parse CLUSTER_LOGIN into user@host
REMOTE_FULL="$CLUSTER_LOGIN"
REMOTE_DIR="$CLUSTER_BASE_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW} Syncing local code to remote server...${NC}"
echo ""
echo "Configuration:"
echo "  Local directory: $LOCAL_DIR"
echo "  Remote: $REMOTE_FULL:$REMOTE_DIR"
echo ""

# Step 1: Rsync local code to remote

rsync -avzh --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*.egg-info/' \
    "$LOCAL_DIR/" "$REMOTE_FULL:$REMOTE_DIR/"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Rsync failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Rsync completed successfully!${NC}"
echo ""



