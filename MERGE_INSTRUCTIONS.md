# Repository Merge Instructions

## Scenario 1: Merge Another Repo Into Aurelius (Preserving Both)

### Prerequisites
- URL of the repository you want to merge
- Decide if you want repos as subdirectories or merged at root level
- Backup your current work: `git branch backup-$(date +%Y%m%d)`

### Method A: Direct Merge (Same Level)

This combines both repos at the root level. **Warning: May cause conflicts if both have similar file structures.**

```bash
# 1. Ensure you're on your working branch
git checkout claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7

# 2. Add the other repository as remote
git remote add other-repo https://github.com/username/other-repository.git

# 3. Fetch all branches and history
git fetch other-repo

# 4. Merge (will need to resolve conflicts)
git merge other-repo/main --allow-unrelated-histories -m "Merge other-repo into Aurelius platform"

# 5. Resolve conflicts
# Open each conflicting file and resolve
git status  # See conflicting files
# Edit files, then:
git add .
git commit -m "Resolve merge conflicts"

# 6. Push to your branch
git push -u origin claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7
```

### Method B: Subdirectory Merge (Recommended - No Conflicts)

This keeps both repos separate in their own directories.

```bash
# 1. Add other repo as subtree
git subtree add --prefix=other-platform https://github.com/username/other-repo.git main --squash

# This creates structure:
# /apps/              (Aurelius platform)
# /other-platform/    (Other repository)
# /compose.yaml       (Aurelius)
# etc.

# 2. Commit and push
git push -u origin claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7
```

### Method C: Reorganize Both Into Subdirectories

Clean separation of both codebases.

```bash
# Step 1: Move Aurelius content to subdirectory
git checkout claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7

# Create new directory for current content
mkdir -p temp-aurelius
git ls-tree --name-only HEAD | grep -v temp-aurelius | xargs -I {} git mv {} temp-aurelius/
git commit -m "Move Aurelius platform to subdirectory"

# Rename to final location
git mv temp-aurelius aurelius-platform
git commit -m "Finalize Aurelius directory structure"

# Step 2: Add other repo
git remote add other-repo https://github.com/username/other-repo.git
git fetch other-repo

# Step 3: Merge with subtree strategy
git merge -s ours --no-commit --allow-unrelated-histories other-repo/main
git read-tree --prefix=other-platform/ -u other-repo/main
git commit -m "Add other platform as subdirectory"

# Final structure:
# /aurelius-platform/  (this codebase)
# /other-platform/     (other codebase)

# Step 4: Push
git push -u origin claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7
```

---

## Scenario 2: Merge Aurelius Into Another Repo

If you want to add Aurelius to another repository:

```bash
# 1. Clone the other repository
git clone https://github.com/username/other-repo.git
cd other-repo

# 2. Add Aurelius as remote
git remote add aurelius https://github.com/alovladi007/Aurelius-Medical-Imaging-Platform.git

# 3. Fetch Aurelius
git fetch aurelius

# 4. Merge as subdirectory
git subtree add --prefix=aurelius-platform aurelius claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7 --squash

# 5. Push to the other repo
git push origin main  # or your target branch
```

---

## Handling Conflicts

When conflicts occur during merge:

```bash
# 1. View conflicting files
git status

# 2. For each conflict:
#    - Open file in editor
#    - Look for conflict markers:
#      <<<<<<< HEAD
#      (your changes)
#      =======
#      (their changes)
#      >>>>>>> other-repo/main
#    - Decide which to keep or combine both
#    - Remove conflict markers

# 3. Stage resolved files
git add <resolved-file>

# 4. Continue merge
git commit -m "Resolve merge conflicts between Aurelius and other-repo"
```

### Common Conflict Files
- `README.md` - Choose one or combine both
- `.gitignore` - Merge both lists
- `package.json` / `requirements.txt` - Merge dependencies
- `docker-compose.yml` - Rename services to avoid conflicts
- `.env.example` - Combine variables

---

## Post-Merge Checklist

After merging, verify:

```bash
# 1. Check all files are present
ls -la

# 2. Verify Docker Compose (if merged at root)
docker compose config  # Should show valid configuration

# 3. Test build (if applicable)
docker compose build

# 4. Check for duplicate dependencies
# - Python: Check multiple requirements.txt files
# - Node: Check multiple package.json files

# 5. Update documentation
# - Update main README.md
# - Document combined architecture
# - Update docker-compose.yml if needed

# 6. Push changes
git push -u origin claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| Merge at root level | `git merge other-repo/main --allow-unrelated-histories` |
| Merge as subdirectory | `git subtree add --prefix=path https://... main --squash` |
| Keep history separate | Use subtree with `--squash` |
| Update subtree later | `git subtree pull --prefix=path https://... main` |
| View merge conflicts | `git status` |
| Abort merge | `git merge --abort` |

---

## Recommendations

**Best Practice:** Use **Method B (Subdirectory Merge)** because:
- ✅ No file conflicts
- ✅ Clear separation of concerns
- ✅ Easy to update either repo independently
- ✅ Preserves both histories
- ✅ Can combine docker-compose files later

**Avoid:** Direct root-level merge unless repos are designed to work together.

---

## Example: Merge Medical Records System

```bash
# Add medical records system as subdirectory
git subtree add --prefix=medical-records https://github.com/org/medical-records.git main --squash

# Update compose.yaml to include both systems
# Edit compose.yaml to add services from medical-records/docker-compose.yml

# Test combined system
docker compose up

# Commit integration
git add compose.yaml
git commit -m "Integrate medical records system with Aurelius platform"
git push -u origin claude/platform-review-analysis-01BSqb7QyV4NoJSdb7t6MUJ7
```

---

## Need Help?

If you encounter issues:
1. Run `git status` to see current state
2. Run `git log --oneline --graph --all` to see branch structure
3. Use `git merge --abort` to cancel a problematic merge
4. Ask for help with specific error messages
