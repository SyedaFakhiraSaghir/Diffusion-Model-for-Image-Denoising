if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "❌ No commits found. Creating initial commit..."
    git add .
    git commit -m "Initial commit: Diffusion Model for Image Denoising"
fi

# Solution 2: Check if master branch exists and push it
if git show-ref --verify --quiet refs/heads/master; then
    echo "✅ Found master branch. Pushing master to origin..."
    git push -u origin master
    exit 0
fi

# Solution 3: If no branches exist, create main
echo "🔄 No branches found. Creating main branch..."
git checkout -b main

# Add all files and commit
echo "💾 Adding files to commit..."
git add .gitignore *.py *.sh

echo "📦 Creating commit..."
git commit -m "feat: Complete diffusion denoising implementation

- Neural network for noise prediction
- MNIST dataset processing with Gaussian noise
- Training loop with validation
- PSNR metrics and visualization
- Environment setup scripts"

# Solution 4: Push with different approaches
echo "🚀 Attempting to push to GitHub..."

# Try pushing main branch
if git push -u origin main; then
    echo "✅ Successfully pushed main branch!"
elif git push -u origin master; then
    echo "✅ Successfully pushed master branch!"
else
    echo "❌ Push failed. Trying force push..."
    git push -u origin main --force
fi

echo "✅ Git issues resolution complete!"