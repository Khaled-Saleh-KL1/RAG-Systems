# Libraries Imports
import os

# Files Imports
from helpers import get_settings

class BaseController:
    def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_dir = os.path.join(os.path.dirname(self.base_dir), 'assets', 'files')
