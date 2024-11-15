import glob

def dir_all_images( path_data):
    extensions = [".jpg", ".png", ".JPG", ".JPEG", ".PNG"]
    images=[]
    for ext in extensions:
        images += glob.glob(f'{path_data}/*{ext}')
    return images