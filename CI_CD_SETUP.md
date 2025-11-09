# CI/CD Setup Guide

This guide explains how to set up automated testing, coverage reporting, and Docker image publishing for the Ollama Proxy project.

## Overview

Two GitHub Actions workflows have been added:

1. **Test and Coverage** (`.github/workflows/test-and-coverage.yml`) - Runs on every push to main and PRs
2. **Docker Build and Push** (`.github/workflows/docker-build-push.yml`) - Builds multi-arch images and pushes to Harbor

## Prerequisites

### For GitHub Actions

You'll need to configure the following secrets in your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

## Required Secrets

### For Test & Coverage (All Optional)

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `CODECOV_TOKEN` | (Optional) Codecov integration token | Sign up at [codecov.io](https://codecov.io) and get your repo token |
| `GIST_SECRET` | (Optional) GitHub PAT for coverage badge | Create at Settings → Developer settings → Personal access tokens (classic) with `gist` scope |
| `GIST_ID` | (Optional) Gist ID for badge storage | Create a public gist, copy the ID from the URL |

**Note:** The test workflow will run successfully without these secrets. They're only needed if you want coverage badges and Codecov integration. Missing secrets will be skipped gracefully.

### For Docker Publishing (Harbor)

| Secret Name | Description | Example |
|------------|-------------|---------|
| `HARBOR_USERNAME` | Harbor username | `admin` or your user |
| `HARBOR_PASSWORD` | Harbor password or robot token | Use robot account for better security |

**Note:** The workflow is pre-configured to push to `container.hect.dev/affine/ollama-proxy`

## Setup Instructions

### 1. Enable GitHub Actions

1. Go to your repository on GitHub
2. Navigate to **Settings → Actions → General**
3. Under "Actions permissions", select **Allow all actions and reusable workflows**
4. Under "Workflow permissions", select **Read and write permissions**
5. Enable **Allow GitHub Actions to create and approve pull requests**

### 2. Configure Harbor

1. Log in to your Harbor instance at `container.hect.dev`
2. Make sure the `affine` project exists
3. Create a **Robot Account** (recommended):
   - Go to Project → Robot Accounts → New Robot Account
   - Give it a name (e.g., `github-actions`)
   - Set expiration and permissions (Push artifact, Pull artifact)
   - Copy the token
4. Add secrets to GitHub:
   - `HARBOR_USERNAME`: Robot account name (e.g., `robot$github-actions`)
   - `HARBOR_PASSWORD`: Robot account token

### 3. Configure Coverage Badge (Optional)

To display a coverage badge in your README:

1. Create a Personal Access Token:
   - Go to [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
   - Click **Generate new token (classic)**
   - Give it a name (e.g., "Coverage Badge")
   - Select only the `gist` scope
   - Generate and copy the token
   - Add as `GIST_SECRET` in repository secrets

2. Create a Gist for badge data:
   - Go to [gist.github.com](https://gist.github.com)
   - Create a new **public** gist
   - Name it `ollama-proxy-coverage.json`
   - Add dummy content: `{}`
   - Copy the gist ID from the URL (e.g., `abc123def456`)
   - Add as `GIST_ID` in repository secrets

3. Add badge to README.md:
   ```markdown
   ![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/YOUR_USERNAME/YOUR_GIST_ID/raw/ollama-proxy-coverage.json)
   ```

### 4. Configure Codecov (Optional)

For detailed coverage reports:

1. Sign up at [codecov.io](https://codecov.io) with your GitHub account
2. Add your repository
3. Copy the repository upload token
4. Add as `CODECOV_TOKEN` in GitHub secrets
5. Codecov will comment on PRs with coverage changes

## Workflow Behavior

### Test and Coverage Workflow

**Triggers:**
- Push to `main` branch
- Pull requests to `main`

**Actions:**
- Installs Python 3.11 and dependencies
- Runs pytest with coverage (90% minimum enforced)
- Uploads coverage to Codecov (if configured)
- Generates coverage badge on main branch (if configured)
- Comments on PRs with coverage report (if configured)
- Uploads HTML coverage report as artifact (main branch only)

**Important:** This workflow runs independently and doesn't block Docker builds.

### Docker Build and Push Workflow

**Triggers:**
- Push to `main` branch → builds `latest` and auto-versioned tags (e.g., `0.1.123`)
- Push tags matching `v*.*.*` → builds version tags (e.g., `v1.0.0`, `1.0`, `1`)
- Pull requests → runs tests only (no Docker build)

**Actions:**
- Runs full test suite first (must pass before building)
- Generates auto-incrementing version tag using GitHub run number (format: `0.1.RUN_NUMBER`)
- Builds multi-architecture images (amd64, arm64)
- Pushes to Harbor registry
- Uses layer caching for faster builds

**Important:** 
- Docker build only happens on push to `main` (not on PRs)
- Tests must pass before Docker images are built
- Each successful main branch build gets a unique auto-incrementing version tag
- Version numbers persist across the lifetime of the repository (never reset)

**Generated Tags (on main branch):**
- `latest` - Always points to the latest main branch build
- `0.1.X` - Auto-incrementing version (X = GitHub Actions run number)
- `sha-<commit>` - Specific commit SHA for traceability

**Generated Tags (on version tags like v1.2.3):**
- `v1.2.3` - Full semantic version
- `1.2` - Major.minor version
- `1` - Major version

## Usage Examples

### Pull from Harbor

```bash
docker pull container.hect.dev/affine/ollama-proxy:latest
docker run -p 8080:8080 --env-file .env container.hect.dev/affine/ollama-proxy:latest

# Or pull a specific version
docker pull container.hect.dev/affine/ollama-proxy:0.1.123
```

### Using docker-compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ollama-proxy:
    image: container.hect.dev/affine/ollama-proxy:latest
    ports:
      - "8080:8080"
    environment:
      - UPSTREAM=http://host.docker.internal:11434
      - PROXY_API_KEY=your-secure-key
      - PROXY_BASE_MODEL=gpt-oss:20b
      - PROXY_TITLE_MODEL=granite3.1-moe:1b
    restart: unless-stopped
```

## Triggering Builds

### Automatic (Main Branch)

Simply push to main:
```bash
git add .
git commit -m "Update feature"
git push origin main
```

### Semantic Version Release

Create and push a tag:
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

This will create images tagged as:
- `v1.0.0`
- `1.0`
- `1`

For main branch pushes, you'll get:
- `latest`
- `0.1.X` (where X is the run number)
- `sha-<commit>`

## Monitoring

### View Workflow Runs

Go to your repository → **Actions** tab to see all workflow runs, logs, and artifacts.

### Check Coverage

- **Codecov**: Visit `https://codecov.io/gh/YOUR_USERNAME/REPO_NAME`
- **Artifacts**: Download HTML coverage report from workflow run artifacts
- **Badge**: Shows in README (if configured)

### Verify Docker Images

- **Harbor**: `https://container.hect.dev/harbor/projects/affine/repositories/ollama-proxy`

## Troubleshooting

### Tests Fail

Check the Actions tab for detailed logs. Common issues:
- Missing dependencies (check `pyproject.toml`)
- Coverage below 90% threshold
- Python version mismatch

### Docker Build Fails

Common issues:
- Missing secrets (check Settings → Secrets)
- Registry authentication errors (verify credentials)
- Network issues (retry the workflow)

### Harbor-Specific Issues

- **401 Unauthorized**: Check robot account permissions on the `affine` project
- **404 Project Not Found**: Ensure the `affine` project exists in Harbor
- **Certificate errors**: If using self-signed certs, may need additional setup

## Minimal Setup (Just Tests)

If you only want automated tests without Docker publishing:

1. Only configure `CODECOV_TOKEN` (optional)
2. The test workflow will run on every push and PR
3. Docker workflow will build but won't push without Harbor credentials

## Advanced: Self-Hosted Runners

For private Harbor instances or faster builds, use self-hosted runners:

1. Go to Settings → Actions → Runners
2. Click **New self-hosted runner**
3. Follow installation instructions
4. Modify workflow files to use: `runs-on: self-hosted`

## Security Notes

- Never commit secrets to the repository
- Use Robot Accounts for Harbor (not personal accounts)
- Use Docker Hub access tokens (not passwords)
- Set appropriate expiration dates for tokens
- Use pull request reviews before merging to main
- Consider using GitHub Environments for additional protection

## Next Steps

- [ ] Configure required secrets in GitHub
- [ ] Push a commit to main to trigger first workflow
- [ ] Verify tests pass and coverage is reported
- [ ] Check Docker images are published to registries
- [ ] Add coverage badge to README
- [ ] Create first release tag (`v1.0.0`)
- [ ] Set up branch protection rules (optional)
