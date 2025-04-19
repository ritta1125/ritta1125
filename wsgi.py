import sys

# Add your project directory to the Python path
path = '/home/ritta1125/mysite'
if path not in sys.path:
    sys.path.append(path)

# Import your Flask app
from app import app as application 