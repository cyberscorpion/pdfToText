from django.shortcuts import render
from .permissions import *
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import os
from glob import glob
import numpy as np
import nltk
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
#if __name__ == '__main__':
#    pdfConverter = PdfConverter(file_path='Rishav.pdf')
 #   print(pdfConverter.save_convert_pdf_to_txt())
    
    
class GET_APIView(APIView):

    permission_classes = (permissions.AllowAny,)
    def get(self, request, format=None):
        
        pdfConverter = PdfConverter(file_path = 'http://www.africau.edu/images/default/sample.pdf')
        print(pdfConverter)
#        pdfConverter = PdfConverter(file_path = reading_file)
#        text = pdfConverter.convert_pdf_to_txt()
#        pdfConverter.save_convert_pdf_to_txt()
#        
#        tokens = word_tokenize(text)
#        
#        tokens = [w.lower() for w in tokens]
#        
#        stop_words = stopwords.words('english')
#        tokens = [word for word in tokens if not word in stop_words]
#        
#        tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
#        tokens = [c for c in tokens if c]
#        
#        y = ['●']
#        x = []
#        for a in tokens:
#            if not a in y and not a in x:
#                x.append(a)
#        
#        tokens = x
#     
#        start = tokens.index("skills")
#        end = tokens.index("experience")
#        skills = tokens[start+1:end]
#        
#        space = " "
#        additional = ['development','programming']
#        for i in range(0,len(skills)):
#            if skills[i] in additional:
#                skills[i-1] = skills[i-1] +space + skills[i]
#                skills[i] = "*"
#        
#        l=[]
#        for x in skills:
#            if x != '*':
#                l.append(x)
#        
#        skills = l
#        
#            
#        lmtzr = WordNetLemmatizer()
#        for i in range(len(skills)):
#            skills[i] = lmtzr.lemmatize(skills[i])
#        
#        #s = ",".join(s for s in skills)
#        
#        name = ""
#        for i in range(len(reading_file)):
#            if reading_file[i] == '.':
#                break
#            else:
#                name += reading_file[i]
#        print("name of file being read = ",name)
#        
#        
#        tagged = nltk.pos_tag(skills)
#        
#        nv = ['NN',"NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"]
#        s = []
#        for i in range(0,len(tagged)):
#            if tagged[i][1] in nv:
#                s.append(tagged[i][0])
#        
#        skills = s
#        print("Skills = ",skills)
#        #headers=["name","skills","fdjbv","khdgf"]
#        af = [name]
#        for i in skills:
#            af.append(i)
#        p.append(af)
#    
#        my_df = pd.DataFrame(p)
#        with open('my_csv.csv','a') as f:
#            my_df.to_csv(f,header=True)
#
#        df=pd.read_csv("my_csv.csv")


        return Response("Rajat")