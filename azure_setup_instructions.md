# Azure Setup Instructions for AI Platform Trainer

This document provides instructions for setting up Azure resources for the AI Platform Trainer project.

## Prerequisites

- Azure subscription
- Azure CLI installed and configured
- GitHub account with access to the repository

## Resource Group Setup

A resource group has been created to organize all Azure resources for this project:

```
Resource Group: ai-platform-rg
Location: East US
```

## Service Principal

A service principal has been created for GitHub Actions to interact with Azure resources:

```
Name: ai-platform-github-actions
Role: Contributor
Scope: /subscriptions/5da8b2e5-23c5-4050-817c-15618778fbfe/resourceGroups/ai-platform-rg
```

## Setting Up GitHub Secrets

The following GitHub secrets need to be set up:

1. **AZURE_CREDENTIALS**: The service principal credentials in JSON format:

```json
{
  "clientId": "0f27c3ce-bff2-4469-8b6a-649eed93618d",
  "clientSecret": "<REDACTED>",
  "subscriptionId": "5da8b2e5-23c5-4050-817c-15618778fbfe",
  "tenantId": "808b31a7-6937-4142-b032-cfec1b35815b",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

Replace `<REDACTED>` with the actual client secret.

2. **PYPI_API_TOKEN**: Your PyPI token for package publishing.

## Resource Creation Steps

### 1. Azure Container Registry

To create a container registry:

```bash
az acr create --resource-group ai-platform-rg --name aiPlatformRegistry --sku Basic
```

### 2. Azure Container App

Set up the Azure Container App:

```bash
# Create Container App Environment
az containerapp env create \
  --resource-group ai-platform-rg \
  --name ai-platform-env \
  --location eastus

# Create Container App
az containerapp create \
  --resource-group ai-platform-rg \
  --name ai-platform-trainer-app \
  --environment ai-platform-env \
  --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
  --target-port 80 \
  --ingress external
```

### 3. Azure Machine Learning Workspace

For GPU-accelerated training:

```bash
# Create ML Workspace
az ml workspace create \
  --resource-group ai-platform-rg \
  --name ai-platform-ml-workspace \
  --location eastus

# Create compute cluster with GPU support
az ml computetarget create amlcompute \
  --name gpu-cluster \
  --vm-size Standard_NC6 \
  --resource-group ai-platform-rg \
  --workspace-name ai-platform-ml-workspace \
  --min-nodes 0 \
  --max-nodes 4 \
  --idle-seconds-before-scaledown 600
```

## Using the Resources

These resources are used by the GitHub Actions workflows:

- `.github/workflows/ci.yml`: CI pipeline for testing and building packages
- `.github/workflows/azure-deploy.yml`: Deployment pipeline for Azure resources

The workflows will automatically deploy to Azure when code is pushed to the main branch or when releases are created.

## Manual Deployments

For manual deployments or testing:

```bash
# Build and push container
az acr build --registry aiPlatformRegistry --image ai-platform-trainer:latest .

# Update container app
az containerapp update \
  --resource-group ai-platform-rg \
  --name ai-platform-trainer-app \
  --image aiPlatformRegistry.azurecr.io/ai-platform-trainer:latest
```

## Security Notes

- The service principal credentials should be kept secure and not committed to source control
- Rotate the client secret periodically for enhanced security
- Restrict the service principal permissions to only what is necessary for deployments
