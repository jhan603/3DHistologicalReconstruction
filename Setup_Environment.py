import os

# To Run the Script in console
# 1. Navigate to where script is
# 2. python Setup_Environment.py

Working_Directory = "D:\\Part4\\700AB\\3DHistologicalReconstruction" # When working at home
#Working_Directory = " "                            # Root directory when working at university

# Use os.chdir() to change the working directory for the Python script
os.chdir(Working_Directory + "\\H653A_11.3")

# Replace 'p4penv' with your preferred environment name
environment_name = 'p4penv'
# Execute the 'conda create' command with os.system()
os.system(f'conda create -n {environment_name} python=3.10')
os.system(f'conda activate {environment_name}')

# Install Required Packages
# Replace 'requirements.txt' with the path to your requirements file
requirements_file = Working_Directory + "\\requirements.txt"

# Execute the 'pip install' command with os.system()
os.system(f'pip install -r {requirements_file}')

# Open VSC
os.system(f'code .')