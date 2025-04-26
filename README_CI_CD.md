# CI/CD Pipeline Documentation

This document explains the Continuous Integration/Continuous Deployment (CI/CD) pipeline set up for the AI Platform Trainer project.

## Overview

The CI/CD pipeline consists of two main workflows:
1. **CI/CD Pipeline** (GitHub Actions) - For testing, building, and publishing packages
2. **Azure Deployment** (GitHub Actions + Azure) - For deploying to Azure Container Apps and setting up GPU environments

## GitHub CI/CD Pipeline

The CI/CD pipeline is implemented using GitHub Actions and is configured to automatically run whenever:

- Code is pushed to the `main` branch
- A pull request is created targeting the `main` branch
- A new release is created
- The workflow is manually triggered

### Pipeline Components

The pipeline consists of the following jobs:

#### 1. Lint Code

This job performs static code analysis to identify potential issues:

- Uses Python 3.9
- Installs project dependencies
- Runs various linters (flake8, black, isort, mypy)

#### 2. Run Tests

This job runs the test suite across multiple Python versions:

- Tests on Python 3.8, 3.9, and 3.10
- Installs project dependencies
- Runs pytest on all tests in the `tests/` directory
- Generates a code coverage report
- Uploads the report to Codecov

#### 3. Build Package

This job builds the Python package:

- Runs after tests pass
- Fetches the complete Git history for proper versioning
- Uses Python 3.9
- Builds both wheel and source distribution using the Python build system
- Archives the resulting packages as GitHub workflow artifacts

#### 4. Publish to PyPI

This job is triggered only when a new release is created:

- Downloads the previously built package artifacts
- Publishes the packages to PyPI using a secure API token
- Skips existing versions to prevent errors

## Azure Deployment Pipeline

The Azure Deployment workflow is configured to run:

- When code is pushed to the `main` branch
- When manually triggered

### Deployment Components

#### 1. Deploy to Azure

This job handles container deployment:

- Runs tests to ensure code quality
- Logs in to Azure using credentials stored as GitHub secrets
- Builds a container image in Azure Container Registry (ACR)
- Updates the Azure Container App with the new image

#### 2. Setup GPU Environment

This job configures GPU resources:

- Creates or updates an Azure ML workspace
- Sets up a GPU-enabled compute cluster with auto-scaling
- Uploads training data to the ML workspace datastore

## Setting Up CI/CD

### Prerequisites

1. **GitHub Repository** - Your code should be in a GitHub repository
2. **Azure Account** - You need an active Azure subscription
3. **PyPI Account** - For publishing packages

### Configuration Steps

#### GitHub Actions Setup

1. The workflow files are already configured in your repository:
   - `.github/workflows/ci.yml`
   - `.github/workflows/azure-deploy.yml`

2. Add required GitHub secrets:
   - `PYPI_API_TOKEN` - For publishing to PyPI
   - `AZURE_CREDENTIALS` - For Azure deployments

#### Creating Azure Credentials

To set up the required Azure credentials:

```bash
# Log in to Azure CLI
az login

# Create a service principal and get credentials
az ad sp create-for-rbac --name "ai-platform-github-actions" \
                         --role contributor \
                         --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
                         --sdk-auth
```

Store the JSON output as a GitHub secret named `AZURE_CREDENTIALS`.

#### Setting Up PyPI Token

1. Generate a PyPI API token at https://pypi.org/manage/account/token/
2. Add the token to your GitHub repository secrets as `PYPI_API_TOKEN`

## Best Practices

For optimal use of this CI/CD pipeline:

1. **Version Management**: Follow semantic versioning (X.Y.Z) when creating new releases
2. **Pull Requests**: Ensure all PRs pass the CI checks before merging
3. **Test Coverage**: Add tests for new features to maintain good coverage
4. **Pre-commit Hooks**: Run pre-commit hooks locally before pushing to catch common issues early

## Troubleshooting

If a workflow fails:

- Check the GitHub Actions logs for detailed error information
- Ensure all required dependencies are correctly specified
- Verify that test configurations match the environment variables available in the workflow
- For Azure-specific issues, check the Azure portal for additional logs and diagnostics

## GPU Testing and Training

The project is configured to support GPU acceleration both locally and in the cloud:

- Local development can use CUDA if available
- The Azure ML workspace has a GPU-enabled compute cluster for training

To enable GPU testing in CI:
1. Set up self-hosted runners with GPU capabilities
2. Uncomment the GPU testing section in the CI workflow file
