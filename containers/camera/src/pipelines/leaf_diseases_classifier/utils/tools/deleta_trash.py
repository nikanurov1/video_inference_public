import subprocess


def delete_trash():
    result = subprocess.run(["find", ".", "-name", ".DS_Store", "-delete"], stderr=subprocess.PIPE, text=True)
    print(result.stderr)
