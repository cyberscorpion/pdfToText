import pandas as pd
import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.model_selection import cross_validate,KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys
import csv

warnings.simplefilter("error")

users=20
items=40
req_sim=0

def readingFile(filename):
    f = open(filename,"r")
    data=[]
    ct=0
    for row in f:
        if ct==0:
            ct=1
            continue
        else:
            #print("row = ",row)
            r = row.split(',')
            #print(r)
            #print(r[0],"\t",type(r[0]))
            #print(r[1],"\t",type(r[1]))
            r[2] = r[2][:-2]
            #print(r[2],"\t",type(r[2]))
            e = [ int(r[0].strip(' " ')) , int(r[1]) , int(r[2]) ]
            data.append(e)
    f.close()
    return data

def similarity_user(data):
    print("Hello user")
    user_similarity_cosine = np.zeros((users,users))
    user_similarity_jaccard = np.zeros((users,users))
    user_similarity_pearson = np.zeros((users,users))
    for user1 in range(users):
        #print(user1)
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

            print(str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_cosine) + "\t" + str(pred_jaccard) + "\t" + str(pred_pearson))
            pred_rate_cosine.append(pred_cosine)
            pred_rate_jaccard.append(pred_jaccard)
            pred_rate_pearson.append(pred_pearson)

        rmse_cosine.append(sqrt(mean_squared_error(true_rate,pred_rate_cosine)))
        rmse_jaccard.append(sqrt(mean_squared_error(true_rate,pred_rate_jaccard)))
        rmse_pearson.append(sqrt(mean_squared_error(true_rate,pred_rate_pearson)))

        print(str(sqrt(mean_squared_error(true_rate,pred_rate_cosine))) + "\t" + str(sqrt(mean_squared_error(true_rate,pred_rate_jaccard))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_pearson))))

    rmse_cosine = sum(rmse_cosine)/float(len(rmse_cosine))
    rmse_pearson = sum(rmse_pearson)/float(len(rmse_pearson))
    rmse_jaccard = sum(rmse_jaccard)/float(len(rmse_jaccard))

    print(str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson))

    f_rmse = open("rmse_user.txt","w")
    f_rmse.write(str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

    rmse = [rmse_cosine , rmse_jaccard , rmse_pearson]

    req_sim = rmse.index(min(rmse))

    print('req_sim = ')
    print(req_sim)
    f_rmse.write(str(req_sim))
    f_rmse.close()

    if req_sim == 0:
        print('sim_mat_user = cosine')
        sim_mat_user = sim_user_cosine
    if req_sim == 1:
        print('sim_mat_user = jaccard')
        sim_mat_user = sim_user_jaccard
    if req_sim == 2:
        print('sim_mat_user = pearson')
        sim_mat_user = sim_user_pearson

    return Mat , sim_mat_user

def predictRating(recommend_data):
    data , sim_user = crossValidation(recommend_data)
   # print(data)
    f = open("toBeRated.csv","r")
    toBeRelated =np.zeros((items),dtype=np.float64)
    user_id = 0
    ct=0
    for row in f:
        if ct>0:
           # print("row = ",row)
            x = int(row)
            toBeRelated[x-1] = 1.0
            #user = int(r[0][1:3])
        ct = ct+1
    f.close()
    fw_w = open('result.csv' , 'w')

    user_similarity = np.zeros((users),dtype=np.float64)

    for user1 in range(users):
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
    print('user_similarity')
    print(user_similarity)
    print('similar users ::: ')
    l = user_similarity.argsort()[-10:][::-1]
    print(l)
    f = open("worked.csv","r")
    job = []
    for row in f:
        a = row.split('\t')
        a = a[0].split(',')
        a[len(a)-1] = a[len(a)-1][:-2]
        job.append(a)
    new_job=[]
    for j in job:
        #print(j)
        x = []
        for i in j:
            if i != ' ':
                x.append(i)
        new_job.append(x)
    f.close()
    job=[]
    for i in range(len(new_job)):
        #print(type(new_job[i]),new_job[i])
        x =[]
        for j in range(len(new_job[i])):
            if new_job[i][j] != '':
                x.append(new_job[i][j])
                #print(new_job[i][j],end='=')
        job.append(x)
        #print()
   # print('---------------------------------------------------------------------------')

    #for i in job:
     #   print(i)

    #print('---------------------------------------------------------------------------')
    #print("Vacancies you can apply for :: ")
    #for i in l:
     #   print(job[i])
    print('--------------------------------------------------------------------------')
    vacancy = []
    for i in l:
        for j in job[i]:
            if not j in vacancy:
                vacancy.append(j)

    #print("Job offers :: ")
    #print(vacancy)

    fw_w.close()
    v = []
    for i in range(len(vacancy)):
        if len(vacancy[i])==4:
            v.append(vacancy[i])
    print("Job offers :: ")
    print(v)
    return v

def recommend_job(user_skill_number):
    df = pd.DataFrame(user_skill_number)
    df.to_csv("toBeRated.csv",index=False)
    recommend_data = readingFile("aman_rating.csv")
    return predictRating(recommend_data)
    
