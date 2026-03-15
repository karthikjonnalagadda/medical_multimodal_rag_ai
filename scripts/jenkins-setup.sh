#!/bin/bash
# Jenkins Deployment Script
# This script sets up Jenkins with all necessary plugins and credentials

set -e

JENKINS_URL="http://localhost:8080"
JENKINS_USER="${1:-admin}"
JENKINS_PASSWORD="${2:-admin}"
GITHUB_TOKEN="${3:-}"
DOCKER_USERNAME="${4:-}"
DOCKER_PASSWORD="${5:-}"

echo "🚀 Starting Jenkins Setup Script"

# Wait for Jenkins to be ready
echo "⏳ Waiting for Jenkins to be ready..."
for i in {1..60}; do
    if curl -f -s "$JENKINS_URL" > /dev/null; then
        echo "✅ Jenkins is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Jenkins did not start in time"
        exit 1
    fi
    sleep 5
done

# Get initial admin password (if first run)
if [ -f ".jenkins_initial_password" ]; then
    INITIAL_PASSWORD=$(cat .jenkins_initial_password)
    echo "Setting up Jenkins with initial admin password..."
fi

# Install recommended plugins using Jenkins CLI
echo "📦 Installing plugins..."
java -jar jenkins-cli.jar -s "$JENKINS_URL" -auth "$JENKINS_USER:$JENKINS_PASSWORD" \
    install-plugin \
    pipeline \
    github \
    github-api \
    email-ext \
    docker-plugin \
    docker-commons \
    kubernetes \
    slack \
    ssh-agent \
    timestamper \
    AnsiColor \
    blueocean \
    blueocean-github-pipeline \
    -restart

# Create credentials
if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
    echo "🔐 Adding Docker credentials..."
    
    CREDENTIAL_JSON=$(cat <<EOF
{
  "": "0",
  "credentials": {
    "scope": "GLOBAL",
    "id": "docker-registry",
    "username": "$DOCKER_USERNAME",
    "password": "$DOCKER_PASSWORD",
    "description": "Docker Hub Credentials",
    "\$class": "com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl"
  }
}
EOF
)
    
    curl -X POST "$JENKINS_URL/credentials/store/system/domain/_/createCredentials" \
        -u "$JENKINS_USER:$JENKINS_PASSWORD" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "json=$CREDENTIAL_JSON" || true
    
    echo "✅ Docker credentials added"
fi

# Create GitHub credentials
if [ -n "$GITHUB_TOKEN" ]; then
    echo "🔐 Adding GitHub credentials..."
    
    CREDENTIAL_JSON=$(cat <<EOF
{
  "": "0",
  "credentials": {
    "scope": "GLOBAL",
    "id": "github-credentials",
    "username": "github-bot",
    "password": "$GITHUB_TOKEN",
    "description": "GitHub Token",
    "\$class": "com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl"
  }
}
EOF
)
    
    curl -X POST "$JENKINS_URL/credentials/store/system/domain/_/createCredentials" \
        -u "$JENKINS_USER:$JENKINS_PASSWORD" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "json=$CREDENTIAL_JSON" || true
    
    echo "✅ GitHub credentials added"
fi

echo ""
echo "✅ Jenkins setup complete!"
echo ""
echo "📊 Access Jenkins at: $JENKINS_URL"
echo "👤 Username: $JENKINS_USER"
echo "🔑 Password: (enter your password)"
echo ""
echo "Next steps:"
echo "1. Create a new Pipeline job"
echo "2. Point it to the Jenkinsfile in the repository"
echo "3. Configure GitHub webhook (optional)"
echo "4. Run your first build!"
