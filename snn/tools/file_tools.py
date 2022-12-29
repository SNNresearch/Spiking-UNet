import os

def check_exist_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
