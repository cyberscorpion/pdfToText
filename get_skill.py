from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import warnings
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import os
from glob import glob
import nltk
import requests
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


class PdfConverter:

    def __init__(self, file_path):
        self.file_path = file_path
        # convert pdf file to a string which has space among words
    def convert_pdf_to_txt(self):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'  # 'utf16','utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(self.file_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)
        fp.close()
        device.close()
        str = retstr.getvalue()
        retstr.close()
        return str
        # convert pdf file text to string and save as a text_pdf.txt file
    def save_convert_pdf_to_txt(self):
        content = self.convert_pdf_to_txt()
        txt_pdf = open('text_pdf.txt', 'wb')
        txt_pdf.write(content.encode('utf-8'))
        txt_pdf.close()



def get_skills(url):
    chunk_size = 2000
    #url = 'http://www.hrecos.org//images/Data/forweb/HRTVBSH.Metadata.pdf'
    #url = 'https://github.com/rajatjain1998/pdfToText/raw/master/Rishav.pdf'

    r = requests.get(url, stream=True)

    print('downloading......')

    with open('AMAN.pdf', 'wb') as fd:
        for chunk in r.iter_content(chunk_size):
            fd.write(chunk)

    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.pdf')]
    #print(files)

    list_of_files = []
    for x in files:
        list_of_files.append(x)

    for x in list_of_files:
        print(x)

    my_df = pd.DataFrame()#columns=["name","skills"])
    my_df.to_csv('my_csv.csv',index=True,header=True)

    read_files = []
    p = []
    for reading_file in list_of_files:
        if reading_file not in read_files:

            read_files.append(reading_file)

            pdfConverter = PdfConverter(file_path = reading_file)
            text = pdfConverter.convert_pdf_to_txt()
            pdfConverter.save_convert_pdf_to_txt()

            tokens = word_tokenize(text)

            tokens = [w.lower() for w in tokens]
            stop_words = set(stopwords.words('english'))
            stop_words.update(('know','learn','work'))
            tokens = [word for word in tokens if not word in stop_words]

            tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
            tokens = [c for c in tokens if c]

            y = ['●']
            x = []
            for a in tokens:
                if not a in y and not a in x:
                    x.append(a)

            tokens = x

            start = tokens.index("skills")
            end = tokens.index("experience")
            skills = tokens[start+1:end]
            space = " "
            additional = ['development','programming']
            for i in range(0,len(skills)):
                if skills[i] in additional:
                    skills[i-1] = skills[i-1] +space + skills[i]
                    skills[i] = "*"

            l=[]
            for x in skills:
                if x != '*':
                    l.append(x)

            skills = l
            lmtzr = WordNetLemmatizer()
            for i in range(len(skills)):
                skills[i] = lmtzr.lemmatize(skills[i])

            name = ""
            for i in range(len(reading_file)):
                if reading_file[i] == '.':
                    break
            else:
                name += reading_file[i]
            print("name of file being read = ",name)


            #tagged = nltk.pos_tag(skills)


            print("Skills = ",skills)
        #headers=["name","skills","fdjbv","khdgf"]
            af = []
            for i in skills:
                af.append(i)
            p.append(af)


    my_df = pd.DataFrame(p)
    with open('my_csv.csv','w') as f:
        my_df.to_csv(f,header=True,index=False)

    df=pd.read_csv("my_csv.csv")
    return skills
