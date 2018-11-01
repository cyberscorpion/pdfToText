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
import io
import scipy.stats
import scipy.spatial
from sklearn.model_selection import cross_validate,KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys

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
        
        chunk_size = 2000
        
        response = requests.get(self.file_path)
        
#        with open('Rishav.pdf','wb') as fd:
#            for chunk in r.iter_content(chunk_size):
#                fd.write(chunk)
        
#        fp = open(self.file_path, 'rb')
        with io.BytesIO(response.content) as fp:
            
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
                interpreter.process_page(page)
#        fp.close()
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
    
import requests
class GET_APIView(APIView):

    permission_classes = (permissions.AllowAny,)
    def post(self, request, format=None):
        p=[]
        url=request.POST.get('url')
        pdfConverter = PdfConverter(file_path = url)
        s=url
        name=s.split('/')[-1].split('.')[0]
#        print(pdfConverter)
#        pdfConverter = PdfConverter(file_path = reading_file)
        text = pdfConverter.convert_pdf_to_txt()
#        print(text)
        pdfConverter.save_convert_pdf_to_txt()
#        
        tokens = word_tokenize(text)
        
        tokens = [w.lower() for w in tokens]
        
        stop_words = stopwords.words('english')
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
        
        print("name of file being read = ",name)
        
        
        tagged = nltk.pos_tag(skills)
        
        nv = ['NN',"NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"]
        s = []
        for i in range(0,len(tagged)):
            if tagged[i][1] in nv:
                s.append(tagged[i][0])
        
        skills = s
        print("Skills = ",skills)
        af = [name]
        for i in skills:
            af.append(i)
        p.append(af)
        dictionary={}
        dictionary['name']=name
        dictionary['skills']=skills
#        print(dictionary)
        
        return Response(dictionary)
    

    
import csv
    
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import urllib.request
import codecs
warnings.simplefilter("error")

users = 20
items = 10
req_sim=0
def readingFile(filename):
#    response = requests.get('https://raw.githubusercontent.com/rajatjain1998/pdfToText/master/ratings.csv')
    data=[]
#    with io.BytesIO(response.content) as f:
    url = 'https://raw.githubusercontent.com/rajatjain1998/pdfToText/master/ratings.csv'
    

    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    
#    response = urllib2.urlopen(url)
#    f = open(filename,"r")
#    f = csv.reader(response)
#    f = [row for row in csvfile]
    for row in csvfile:
#        print(line)
#    for row in f:
#        print(row)
#        r = row.split(',')
        r = row.split(',')
        e = [ int(r[0].strip(' " '))%7292024814 , int(r[1]) , int(r[2]) ]
        data.append(e)
    return data

def similarity_user(data):
    print ("Hello user")
    user_similarity_cosine = np.zeros((users,users))
    user_similarity_jaccard = np.zeros((users,users))
    user_similarity_pearson = np.zeros((users,users))
    for user1 in range(users):
        print (user1)
        for user2 in range(users):
            if np.count_nonzero(data[user1]) and np.count_nonzero(data[user2]):
                user_similarity_cosine[user1][user2] = 1 - scipy.spatial.distance.cosine(data[user1],data[user2])
                user_similarity_jaccard[user1][user2] = 1 - scipy.spatial.distance.jaccard(data[user1],data[user2])
                try:
                    if not math.isnan(scipy.stats.pearson(data[user1],data[user2])[0]):
                        user_similarity_pearson[user1][user2] = scipy.stats.pearson(data[user1],data[user2])[0]
                    else:
                        user_similarity_pearson[user1][user2] = 0
                except:
                    user_similarity_pearson[user1][user2] = 0
    return user_similarity_cosine , user_similarity_jaccard , user_similarity_pearson

def crossValidation(data):
    k_fold = KFold(n_splits=3,shuffle=True)
    Mat = np.zeros((users,items))
    for e in data:
        Mat[e[0]-1][e[1]-1] = e[2]

    sim_user_cosine , sim_user_jaccard , sim_user_pearson = similarity_user(Mat)

    rmse_cosine = []
    rmse_jaccard = []
    rmse_pearson = []

    for train_indices , test_indices in k_fold.split(data):
        train = [data[i] for i in train_indices]
        test = [data[i] for i in test_indices]
        M = np.zeros((users,items))
        for e in train:
            M[e[0]-1][e[1]-1] = e[2]

        true_rate = []
        pred_rate_cosine=[]
        pred_rate_jaccard=[]
        pred_rate_pearson=[]

        for e in test:
            user = e[0]
            item = e[1]
            true_rate.append(e[2])

            pred_cosine = 3.0
            pred_jaccard = 3.0
            pred_pearson = 3.0

            if np.count_nonzero(M[user-1]):
                sim_cosine = sim_user_cosine[user-1]
                sim_jaccard = sim_user_jaccard[user-1]
                sim_pearson = sim_user_pearson[user-1]
                ind = ( M[:,item-1] > 0 )
                normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
                normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))
                normal_pearson = np.sum(np.absolute(sim_pearson[ind]))
                if normal_cosine > 0:
                    pred_cosine = np.dot(sim_cosine , M[:,item-1])/normal_cosine
                if normal_jaccard > 0:
                    pred_jaccard = np.dot(sim_jaccard , M[:,item-1])/normal_jaccard
                if normal_pearson > 0:
                    pred_pearson = np.dot(sim_pearson , M[:,item-1])/normal_pearson

            if pred_cosine < 0:
                pred_cosine = 0
            if pred_cosine > 5:
                pred_cosine = 5
            if pred_jaccard < 0:
                pred_jaccard = 0
            if pred_jaccard > 5:
                pred_jaccard = 5
            if pred_pearson < 0:
                pred_pearson = 0
            if pred_pearson > 5:
                pred_pearson = 5

            print (str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_cosine) + "\t" + str(pred_jaccard) + "\t" + str(pred_pearson))
            pred_rate_cosine.append(pred_cosine)
            pred_rate_jaccard.append(pred_jaccard)
            pred_rate_pearson.append(pred_pearson)

        rmse_cosine.append(sqrt(mean_squared_error(true_rate,pred_rate_cosine)))
        rmse_jaccard.append(sqrt(mean_squared_error(true_rate,pred_rate_jaccard)))
        rmse_pearson.append(sqrt(mean_squared_error(true_rate,pred_rate_pearson)))

        print (str(sqrt(mean_squared_error(true_rate,pred_rate_cosine))) + "\t" + str(sqrt(mean_squared_error(true_rate,pred_rate_jaccard))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_pearson))))

    rmse_cosine = sum(rmse_cosine)/float(len(rmse_cosine))
    rmse_pearson = sum(rmse_pearson)/float(len(rmse_pearson))
    rmse_jaccard = sum(rmse_jaccard)/float(len(rmse_jaccard))

    print (str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson))

    f_rmse = open("rmse_user.txt","w")
    f_rmse.write(str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

    rmse = [rmse_cosine , rmse_jaccard , rmse_pearson]

    req_sim = rmse.index(min(rmse))

    print ('req_sim = ')
    print (req_sim)
    f_rmse.write(str(req_sim))
    f_rmse.close()

    if req_sim == 0:
        print ('sim_mat_user = cosine')
        sim_mat_user = sim_user_cosine
    if req_sim == 1:
        print ('sim_mat_user = jaccard')
        sim_mat_user = sim_user_jaccard
    if req_sim == 2:
        print ('sim_mat_user = pearson')
        sim_mat_user = sim_user_pearson

    return Mat , sim_mat_user

def predictRating(recommend_data,user_id,skills):

    data , sim_user = crossValidation(recommend_data)

    #f = open(sys.argv[2] , "r")
#    f = open("toBeRated.csv","r")
    toBeRelated =np.zeros((items),dtype=np.float64)
#    user_id = 0
#    ct=0
    for x in skills:
#        if ct>0:
#            print (row)
#            r = row.split(',')
#            x = int(r[1][0])
            toBeRelated[x-1] = 1.0
#            user = int(r[0][1:3])
#        ct = ct+1
#    f.close()
    """
    print 'toBeRealted = '
    print toBeRelated
    print type(toBeRelated)

    print 'toBeRealted = '
    print toBeRelated

    print type(toBeRelated)
    """

#    print (toBeRelated)
    fw_w = open('result.csv' , 'w')

    user_similarity = np.zeros((users),dtype=np.float64)

    for user1 in range(users):
        #print 'data[user1] = '
        #print data[user1]


        if np.count_nonzero(data[user1]):
            if req_sim == 0:
                user_similarity[user1] = 1.0 - scipy.spatial.distance.cosine(toBeRelated , data[user1])
            else:
                if req_sim == 1:

                    user_similarity[user1] = 1.0 - scipy.spatial.distance.jaccard(toBeRelated,data[user1])
                else:
                    try :
                        if not math.isnan(scipy.stats.prearson(toBeRelated,data[user1])[0]):
                            user_similarity[user1] = scipy.stats.pearson(toBeRelated,data[user1])[0]
                        else:
                            user_similarity[user1] = 0
                    except:
                        user_similarity[user1]  = 0
    print ('user_similarity')
    print (user_similarity)
    print ('similar users ::: ')
    l = user_similarity.argsort()[-5:][::-1]
    for i in l:
        print (i+1)
    f = open("user_job_new.csv","r")
    job = []
    for row in f:
        a=row.split(',')
        a[0]=a[0][1:]
        a[len(a)-1]=a[len(a)-1][:-2]
        job.append(a)
    fw_w.close()
    f.close()
    print (job)
    
    company = []
    print("Companies you can apply for :: ")
    for i in l:
        ct=0
        for j in job[i]:
            if ct>0:
                print(j.rstrip(' " '))
                company.append(j)
            ct=1
    print(company)
    company_list = []
    f = open("company.csv","r")
    for row in f:
        a = row.split(',')
        a[len(a)-1]=a[len(a)-1][:-1]
        company_list.append(a)
    f.close()
    print("Company database ::: ")
    print(company_list)

    candidate_skill = []
    for i in range(len(toBeRelated)):
        if toBeRelated[i]>0:
            candidate_skill.append(i)

    print("Candidate's skill : ", candidate_skill)
    vacancy = []
    for c in company_list:
        if c[1] in company:
            skill = c[2:]
            print("skills required in ",c[1],' = ', skill)
            for i in candidate_skill:
                print("i = ",i)
                if str(i) in skill and not str(i) in vacancy:
                    print("*i = ",i)
                    vacancy.append(str(i))
    return(vacancy)

#recommend_data = readingFile(sys.argv[1])
class job_APIView(APIView):

    permission_classes = (permissions.AllowAny,)
    def post(self, request, format=None):
        user_id=request.POST.get('user_id')
        skills=request.POST.get('skills')
        skills=list(map(int,skills.split(',')))
        
        
        recommend_data = readingFile("ratings.csv")
#        return Response(recommend_data)
        
        return Response({'data':predictRating(recommend_data,user_id,skills)})
        
        
class abc(APIView):
    permission_classes = (permissions.AllowAny,)
    def post(self,request, format = None):
        a=request.POST.get('a')
        a=list(map(float,a.split(',')))
        print(a)
#        if request.POST.get('a'):
#            url=request.POST.get('url')
#
#            print("UE")
#            a= request.POST.get('a')
#            print(a)
        
        return Response({'a':2})
    
