import os
from os import getcwd

classes=['cat','dog']
sets=['train']

if __name__=='__main__':
    wd=getcwd()
    for se in sets:
        list_file=open('cls_'+ se +'.txt','w')

        datasets_path=se
        types_name=os.listdir(datasets_path)#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id=classes.index(type_name)#输出0-1
            photos_path=os.path.join(datasets_path,type_name)
            photos_name=os.listdir(photos_path)
            for photo_name in photos_name:
                _,postfix=os.path.splitext(photo_name)#该函数用于分离文件名与拓展名
                if postfix not in['.jpg','.png','.jpeg']:
                    continue
                list_file.write(str(cls_id)+';'+'%s/%s'%(wd, os.path.join(photos_path,photo_name)))
                list_file.write('\n')
        list_file.close()


