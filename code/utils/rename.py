import os


if __name__=="__main__":
    path="/data/datasets/nefii/ds_physg/coffe_simple_color/test/sp_rgb/"

    file_list=os.listdir(path)

    for file in file_list:
        file_=file.split(".")
        file_[-2]='00'
        file_new='.'.join(file_)
        cmd="mv {} {}".format(os.path.join(path,file),os.path.join(path,file_new))
        print(cmd)
        os.system(cmd)