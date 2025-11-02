# Git Commands Guide for This Project

## Basic Git Workflow

### 1. **Checking Status**
```bash
git status
```
Shows which files are modified, staged, or untracked.

### 2. **Staging Files (Preparing to Commit)**
```bash
# Stage all files
git add .

# Stage specific file
git add path/to/file

# Stage all changes (modified and deleted files)
git add -A
```

### 3. **Committing Changes**
```bash
# Commit with message
git commit -m "Your commit message describing the changes"

# Commit with longer message (opens editor)
git commit
```

### 4. **Pushing to GitHub**
```bash
# Push to main branch
git push origin main

# Push to specific branch
git push origin branch-name
```

### 5. **Pulling from GitHub**
```bash
# Pull latest changes from main branch
git pull origin main

# Pull and merge from specific branch
git pull origin branch-name
```

## Common Workflows

### Setting Up on a New Device

1. **Clone the repository:**
```bash
git clone https://github.com/lance449/cyber-threat-v2.git
cd cyber-threat-v2
```

2. **Set up Python environment:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

3. **Set up Node.js (for frontend):**
```bash
cd frontend
npm install
```

4. **Pull latest changes (if working on existing clone):**
```bash
git pull origin main
```

### Daily Workflow

1. **Before starting work, pull latest changes:**
```bash
git pull origin main
```

2. **Make your changes**

3. **Stage your changes:**
```bash
git add .
```

4. **Commit with descriptive message:**
```bash
git commit -m "Description of what you changed"
```

5. **Push to GitHub:**
```bash
git push origin main
```

### Working with Branches

```bash
# Create a new branch
git checkout -b feature-branch-name

# Switch to a branch
git checkout branch-name

# List all branches
git branch

# Push new branch to GitHub
git push origin branch-name

# Delete a branch
git branch -d branch-name
```

### Checking What Changed

```bash
# See what files changed
git status

# See detailed changes
git diff

# See commit history
git log

# See commit history (compact)
git log --oneline
```

### Undoing Changes

```bash
# Unstage a file (but keep changes)
git reset HEAD file-name

# Discard changes in a file (WARNING: permanent!)
git checkout -- file-name

# Undo last commit (keeps changes)
git reset --soft HEAD~1
```

## Important Notes

- **Always pull before pushing** to avoid conflicts:
  ```bash
  git pull origin main
  git push origin main
  ```

- **Write meaningful commit messages** that describe what you changed

- **Large files** (like model files) are handled by Git LFS automatically

- **Never commit sensitive data** like API keys, passwords, or personal information

## Remote Repository Info

- **Repository URL:** https://github.com/lance449/cyber-threat-v2.git
- **Main Branch:** main
- **Remote Name:** origin

