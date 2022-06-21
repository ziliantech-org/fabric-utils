import fitz
import re
import os

def judge_image(path,file):
    checkXO = r"/Type(?= */XObject)"
    checkIM = r"/Subtype(?= */Image)"
    doc = fitz.open(path+file)
    imgcount = 0
    lenXREF = doc.xref_length()
    for i in range(1, lenXREF):
        # 定义对象字符串
        text = doc.xref_object(i)
        isXObject = re.search(checkXO, text)
        # 使用正则表达式查看是否是图片
        isImage = re.search(checkIM, text)
        # 如果不是对象也不是图片，则continue
        if not isXObject or not isImage:
            continue
        imgcount += 1
    if imgcount>0:
        return '搭配图片或图表  '+file
    else:
        return '纯文字  '+file

if __name__ == '__main__':
    path = "D:\path\"
    dirs = os.listdir( path )
    for file in dirs:
        print(judge_image(path,file))
