# Navigate to your project directory
cd /path/to/your/Diffusion-Model-for-Image-Denoising

# Remove the existing git repository (if corrupted)
rm -rf .git

# Initialize a fresh git repository
git init

# Set your git configuration (replace with your info)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add the proper files
git add .gitignore
git add *.py
git add *.sh

# Make initial commit
git commit -m "Initial commit: Diffusion model for image denoising"

# If you had a remote repository, add it back
git remote add origin https://github.com/SyedaFakhiraSaghir/Diffusion-Model-for-Image-Denoising
git push -u origin main