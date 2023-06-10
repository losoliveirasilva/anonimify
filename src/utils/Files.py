import os
import filetype

class Files():

    def __init__(self, directory="./"):
        self.files = list(os.listdir(directory))

    def video(self):
        return [f for f in [self.files] if f.mime.split("/")[0] == "video"]

    def image(self):
        return [f for f in [self.files] if f.mime.split("/")[0] == "image"]
