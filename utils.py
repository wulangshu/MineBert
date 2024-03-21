import pandas as pd
import numpy as np
import os
import json
import csv
import random
import re
#albert
origin_path='D:\\mine-bert\\wiki_zh_2019\\wiki_zh'
wiki_path='D:\\mine-bert\\wiki_zh_2019\\wiki_zh.csv'
rowlines_path='D:\\mine-bert\\wiki_zh_2019\\wiki_rowlines.csv'

def preprocess(origin_path:str,wiki_path:str):
    file_dir=os.listdir(origin_path)
    file_df=pd.DataFrame(columns=['mid','title','text'])
    for i in file_dir:
        file_name=os.listdir(origin_path+'\\'+i)
        for j in file_name:
            with open(origin_path+'\\'+i+'\\'+j,'r',errors="ignore", encoding="utf-8") as load_lines: 
                file=load_lines.readlines() 
                length=len(file)
                for k in range(length):
                    new_k=json.loads(file[k].strip('\n'))
                    mid=new_k['id']+'\t'
                    title=new_k['title']+'\t'
                    text=new_k['text'].strip(title).replace('\n',r'').replace('\\n','').replace('\\','').replace(r'（）','')+'\t'
                    file_df.loc[k]=([mid,title,text])
    file_df.to_csv(wiki_path)

def getLSline(string:str):#获得长短句
    box=[]
    pattern="[，；！。？、]"
    split_text=re.split(pattern,string)
    cache=''
    for i in split_text:
        if cache == '':
            cache= i
            while len(cache)>29:#截断
                box.append(cache[:29]+'。')
                cache=cache[29:]
            continue
        if random.random()<0.6 and len(cache)+len(i)<29 and i!='':# 依概率得到长短句
            cache+='，'
            cache+=i
            continue
        else:
            box.append(cache+'。')
            cache=i
            while len(cache)>29:#截断
                box.append(cache[:29]+'。')
                cache=cache[29:]
    if(cache!=''):
        box.append(cache)
    return box
    

def getrowlines(wiki_path:str,rowlines_path:str):
    file_old=pd.read_csv(wiki_path)
    write_lines=open(rowlines_path, "w",errors="ignore", encoding="utf-8",newline='')
    w=csv.writer(write_lines)
    w.writerow(['mid','lid','text'])
    lid=0
    for j in range(len(file_old['mid'])):
        mid=file_old['mid'][j]
        text=file_old['text'][j].strip('\t')
        LSline=getLSline(text)
        
        for k in LSline:
            w.writerow([str(mid)+'\t',str(lid)+'\t',k+'\t'])

            lid+=1
    write_lines.close()

preprocess(origin_path,wiki_path)
getrowlines(wiki_path,rowlines_path)




