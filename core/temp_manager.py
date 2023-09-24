import atexit
import os
import tempfile

class TempFileManager:
    _instance = None
    temp_files = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            atexit.register(cls.cleanup)
        return cls._instance

    def create_temp_file(self, suffix):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self.temp_files.append(temp_file.name)
        return temp_file

    @classmethod
    def cleanup(cls):
        for temp_file in cls.temp_files:
            try:
                # Remove the temporary file
                os.remove(temp_file)
            except OSError:
                pass
