# Quick Reference: CI/CD Commands

## Local Testing (Before Push)

```bash
# Run tests with coverage
pytest --cov=ollama_proxy --cov-report=term-missing

# Build Docker image locally
docker build -t ollama-proxy:test .

# Test local Docker image
docker run --rm -p 8080:8080 --env-file .env ollama-proxy:test

# Multi-platform build (like CI does)
docker buildx build --platform linux/amd64,linux/arm64 -t ollama-proxy:test .
```

## Triggering Workflows

```bash
# Regular push (triggers tests, then Docker build with auto-version)
git add .
git commit -m "Your changes"
git push origin main
# This will create tags: latest, 0.1.X, sha-<commit>

# Create a release (triggers tests + Docker build with semantic version tags)
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
# This will create tags: v1.0.0, 1.0, 1

# Push to a feature branch (triggers tests only via PR)
git checkout -b feature/my-feature
git push origin feature/my-feature
# Then create a PR on GitHub - only runs tests, no Docker build
```

## Auto-Versioning

Every push to main generates an auto-incrementing version based on the GitHub Actions run number:

```bash
# First push to main: 0.1.1
# Second push: 0.1.2
# Third push: 0.1.3
# etc.
```

You can pull specific versions:
```bash
docker pull harbor.example.com/project/ollama-proxy:latest
docker pull harbor.example.com/project/ollama-proxy:0.1.42
docker pull harbor.example.com/project/ollama-proxy:sha-abc123
```

## Manually Trigger Workflows

```bash
# Install GitHub CLI
# brew install gh  # macOS
# sudo apt install gh  # Ubuntu

# Trigger test workflow manually
gh workflow run "Test and Coverage"

# Trigger Docker build manually
gh workflow run "Docker Build and Push"

# View workflow runs
gh run list

# View logs for latest run
gh run view --log
```

## Checking Status

```bash
# View all workflow runs
gh run list --workflow="Test and Coverage"

# View specific run
gh run view RUN_ID

# Watch a running workflow
gh run watch

# Download artifacts (e.g., coverage report)
gh run download RUN_ID
```

## Docker Registry Operations

### Harbor

```bash
# Login to Harbor
docker login harbor.example.com

# Pull latest image
docker pull harbor.example.com/project/ollama-proxy:latest

# Pull specific version
docker pull harbor.example.com/project/ollama-proxy:v1.0.0

# Pull auto-versioned build
docker pull harbor.example.com/project/ollama-proxy:0.1.42

# Pull by commit SHA
docker pull harbor.example.com/project/ollama-proxy:sha-abc123def

# List tags via API (requires auth)
curl -u "username:password" \
  https://harbor.example.com/api/v2.0/projects/project/repositories/ollama-proxy/artifacts
```

## Local Development Workflow

## Environment Setup

```bash
# Set GitHub secrets via CLI
gh secret set HARBOR_REGISTRY
gh secret set HARBOR_USERNAME
gh secret set HARBOR_PASSWORD
gh secret set HARBOR_PROJECT

# List current secrets
gh secret list

# Delete a secret
gh secret delete SECRET_NAME
```

## Debugging Failed Workflows

```bash
# View failed run logs
gh run view FAILED_RUN_ID --log

# Re-run failed jobs
gh run rerun FAILED_RUN_ID

# Re-run only failed jobs
gh run rerun FAILED_RUN_ID --failed

# Download artifacts from failed run
gh run download FAILED_RUN_ID
```

## Local Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and test locally
pytest --cov=ollama_proxy --cov-report=term-missing

# 3. Build and test Docker image
docker build -t ollama-proxy:dev .
docker run --rm -p 8080:8080 --env-file .env ollama-proxy:dev

# 4. Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# 5. Create PR (triggers test workflow)
gh pr create --title "Add new feature" --body "Description of changes"

# 6. After PR is merged, tag release
git checkout main
git pull
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
```

## Release Workflow

```bash
# 1. Update version in pyproject.toml if needed
# 2. Commit version bump
git add pyproject.toml
git commit -m "Bump version to 1.1.0"
git push origin main

# 3. Create and push tag
git tag -a v1.1.0 -m "Release version 1.1.0

Changes:
- Feature A
- Feature B
- Bug fix C
"
git push origin v1.1.0

# 4. Create GitHub release (optional, for release notes)
gh release create v1.1.0 --title "Version 1.1.0" --notes "
## Changes
- Feature A
- Feature B  
- Bug fix C

## Docker Images
- \`docker pull harbor.example.com/project/ollama-proxy:v1.1.0\`
- \`docker pull harbor.example.com/project/ollama-proxy:0.1.50\`

## Notes
- Tests run first and must pass before Docker builds
- Each main branch push gets an auto-incrementing version (0.1.X)
- Use semantic version tags (v1.0.0) for official releases
"
```

## Useful GitHub Actions URLs

- Workflow runs: `https://github.com/hectorandac/affine-ollama-proxy/actions`
- Secrets settings: `https://github.com/hectorandac/affine-ollama-proxy/settings/secrets/actions`
- Workflow files: `https://github.com/hectorandac/affine-ollama-proxy/tree/main/.github/workflows`

## Useful Docker URLs

- Harbor UI: `https://harbor.example.com/harbor/projects/PROJECT_ID/repositories/ollama-proxy`

## Common Issues

### "Resource not accessible by integration" Error
- Go to Settings → Actions → General
- Enable "Read and write permissions"
- Enable "Allow GitHub Actions to create and approve pull requests"

### Docker Build Fails with "no space left on device"
- GitHub runners have limited disk space
- Use `--no-cache` flag sparingly
- Ensure .dockerignore excludes unnecessary files

### Coverage Below 90% Threshold
- Add more tests
- Or temporarily adjust threshold in pyproject.toml:
  ```toml
  [tool.coverage.report]
  fail_under = 85
  ```

### Harbor 401 Unauthorized
- Verify robot account has push permissions
- Check HARBOR_USERNAME includes "robot$" prefix
- Ensure HARBOR_PASSWORD is the token, not account password
