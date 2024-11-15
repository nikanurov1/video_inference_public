import os
# from utils.run.keep_answers import Keep_answers
import shutil
import glob


def create_newDirName(out_dir,out_name):
    dst_dir=os.path.join(out_dir,out_name)
    init_out_name=out_name
    
    if os.path.isdir(dst_dir):
        number=0
        while os.path.isdir(dst_dir):
            number+=1
            out_name=f"{init_out_name}_{str(number).zfill(3)}"
            dst_dir=os.path.join(out_dir,out_name)

    assert not os.path.isdir(dst_dir), 'Dir already exist!!! please chose another --out-name'
    os.mkdir(dst_dir)
    print(f"dst_dir {dst_dir}")
    print(f"out_name {out_name}")
    return dst_dir

def create_dir_classes(out_dir,out_name,classes):
    # Cheking
    dst_dir=create_newDirName(out_dir,out_name)

    for name_calss in classes:
        class_dir=os.path.join(dst_dir, name_calss)
        os.mkdir(class_dir)
    return dst_dir


def save_imgs(dst_dir,classes,keep_answers, add_conf2name):
    paths_images, indices_out, confs = keep_answers.get_answers()

    for pth_img, index, conf in zip(paths_images, indices_out, confs):
        name_dir_class= classes[int(index)]
        if add_conf2name:
            dst=conf_name2save(dst_dir,name_dir_class,pth_img,conf)
        else:
            dst=os.path.join(dst_dir,name_dir_class)
        shutil.copy(pth_img,dst)


def conf_name2save(dst_dir,name_dir_class,pth_img,conf):
    base_name=os.path.basename(pth_img)
    extension_file=base_name[base_name.rfind("."):]
    base_name=base_name[:base_name.rfind(".")]
    base_name
    new_name = f"{base_name}_{str(round(float(conf), 3))}{extension_file}"
    dst=os.path.join(dst_dir,name_dir_class,new_name)
    return dst


def make_safely_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


def dir_all_images( path_data):
    extensions = [".jpg", ".png", ".JPG", ".JPEG", ".PNG"]
    images=[]
    for ext in extensions:
        images += glob.glob(f'{path_data}/*{ext}')
    return images
