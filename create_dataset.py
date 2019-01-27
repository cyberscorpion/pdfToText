import csv
from random import randint
import pandas as pd
from collections import defaultdict
import pandas as pd
import numpy as np

job=[]
f = open("company.csv","r")
for row in f:
    a = row.split(',')
    a[len(a)-1] = a[len(a)-1][:-2]
    job.append(a)
f.close()
print("job")
for i in job:
    print(i)
print('--------------------------------------------------------------------------')
jobid = []
jobdesp = []
company=[]
skill=[]
for i in job:
    jobid.append(i[0])
    jobdesp.append(i[1])
    company.append(i[2])
    print("type(i[3:]) = ",type(i[3]))
    x = i[3:]
    x[0] = x[0].strip('"')
    for i in range(len(x)):
        if x[i][:5]=='cloud':
            x[i] = 'cloudcomputing'
    skill.append(x)

for i in range(1,len(jobid),1):
    print(jobid[i],"\t",jobdesp[i],"\t",company[i])

print('---------------------------------------------------------------------------')
print('skills')
for i in range(1,len(jobid),1):
    print(i,skill[i])
#for i in range(len(jobid)):
#    print(jobid[i],"\t",jobdesp[i],"\t",company[i],"\t",skill[i])


s=[]
for i in range(1,len(jobid),1):
    for j in range(len(skill[i])):
        s.append(skill[i][j])

print('---------------------------------------------------------------------------')
print('s')
for i in range(len(s)):
    print(i,s[i])


a = []
for i in range(len(s)):
    if not s[i] in a and s[i] != '':
        a.append(s[i])

print('---------------------------------------------------------------------------')
print('a')
print(a)

df = pd.DataFrame(a)
df.to_csv("unique_skills.csv",index=False,header=False)



print('-------------------------------------------------------------------------------')

dict = defaultdict(int)

for i in a:
    dict[i] = 0

print('----------------------------------------------------------------------------------')

df = pd.DataFrame()
df.to_csv("skill.csv",index=False,header=False)
df.to_csv("worked.csv",index=False,header=False)
df.to_csv("skill_number.csv",index=False,header=False)

no_of_users = 20
n = []
y = []
z = []
print('len(a) = ',len(a))
for user in range(no_of_users):
    no_of_user_skill = randint(1,len(a))
    x = []
    t = []
    worked = []
    for i in range(min(6,no_of_user_skill)):
        p = randint(0,len(a)-1)
        print('p -> ',p)
        while a[p] in x or dict[a[p]]>=3:
            print('p = ',p,end=',')
            p = randint(0,len(a)-1)
        print()
        x.append(a[p])
        t.append(p)
        dict[a[p]] = dict[a[p]]+1
        #print(a[p],'->',dict[a[p]],end=' ')
    #print()
    for i in dict:
        print(i,'->',dict[i],end=',')
    print()
    for i in x:
        for j in range(len(jobid)):
            if i in skill[j] and not jobid[j] in worked:
                worked.append(jobid[j])
    print("user ",user)
    print("skills = ",x,"  len(skills) = ",len(x))
    #print("worked = ",worked)
    y.append(x)
    z.append(worked)
    n.append(t)

df = pd.DataFrame(y)
with open("skill.csv","a") as f:
    df.to_csv(f,index=False,header=False)
f.close()

df = pd.DataFrame(z)
with open("worked.csv","a") as f:
    df.to_csv(f,index=False,header=False)
f.close()

df = pd.DataFrame(n)
with open("skill_number.csv","a") as f:
    df.to_csv(f,index=False,header=False)
f.close()
print('--------------------------------------------------------------------------------------')
print("dict")
print(dict)
