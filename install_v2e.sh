# 1. Activate the venv
python3 -m venv venv
source venv/bin/activate

# 2. Verify you're inside (prompt should show (venv))
which python
# should show something like: /home/dimi/Desktop/Projects/Project3-1/venv/bin/python

# 3. Clone v2e
git clone https://github.com/SensorsINI/v2e.git
cd v2e

# 4. Install dependencies and v2e
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# 5. Test installation
python -c "import v2e; print('v2e imported successfully!')"
