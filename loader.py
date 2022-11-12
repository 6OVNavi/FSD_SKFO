import os
import cv2
import pandas as pd
import re

dir='D:\!!ufa\skfo\ii\WhaleReId2_mm'

df=pd.DataFrame(columns=['image', 'cls', 'type'])

trainarr=[]
valarr=[]


for i in os.listdir(dir):
    for j in os.listdir(dir+"/"+i):

        if j == 'desktop.ini':
            continue

        arr=os.listdir(f"{dir}/{i}/{j}")
        koef=int(len(arr)*0.25)
        cnt = 0
        if arr[koef:][0][-4:]=='.png':
            cnt = -1
        for g in arr[koef:]:
            if g[-4:]=='.jpg' and re.search('rotate', g)==None and cnt%10==0:
                trainarr.append([g, f'/{i}/{j}/{g}', i, 'train_opt'])
            cnt += 1

        cnt=0
        for g in arr[:koef]:
            if g[-4:] == '.jpg' and re.search('rotate', g) == None and cnt%10==0:
                valarr.append([g, f'/{i}/{j}/{g}', i, 'val_opt'])
            cnt += 1



df=pd.DataFrame(valarr, columns=['image', 'dir', 'cls', 'type'])
df=pd.concat([df, pd.DataFrame(trainarr, columns=['image', 'dir', 'cls', 'type'])], axis=0)
df=df.reset_index(drop=True)

print(df)

df.to_csv('df3.csv', index=False)


print(df[df.type=='val_opt'].cls.value_counts().keys())
for i in df[df.type=='val_opt'].cls.value_counts().keys():
    try:
        os.mkdir(f'D:/!!ufa/skfo/val_opt/{i}')
    except:
        pass
for i in df[df.type=='train_opt'].cls.value_counts().keys():
    try:
        os.mkdir(f'D:/!!ufa/skfo/train_opt/{i}')
    except:
        pass

for i in range(len(df.type)):
    img = cv2.imread(f'{dir}{df.dir[i]}')
    os.chdir(rf'D:/!!ufa/skfo/sum/{df.cls[i]}')
    cv2.imwrite(f'{df.dir[i].split("/")[-1]}' , img)

#img=cv2.imread('D:/!!ufa/skfo/crop2_DJI_0066_1.jpg')
#cv2.imshow('name', img)

