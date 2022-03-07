import math
from mip import Model, xsum, minimize, BINARY
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
import pandas as pd
from collections import namedtuple, Counter
import copy

def optimization(df,A,aval,Adomain,klst,kval,alpha,betalst,beta0,target):
    pogivenk=get_prob_o_regression(df,klst,kval,target,[1])

    condition=copy.deepcopy(klst)
    condition.extend(A)
    conditionval=copy.deepcopy(kval)
    conditionval.extend(aval)
    popgivenak=get_prob_o_regression(df,condition,conditionval,target,[0])
    pogivenak=get_prob_o_regression(df,condition,conditionval,target,[1])
    pagivenk=get_prob_o_regression(df,klst,kval,A,aval)

    popagivenk=popgivenak*pagivenk

    alphak=pogivenak+alpha*popgivenak
    print (pogivenk,popagivenk)

    m = Model("Test")
    i=0
    var_lst=[]
    var_map={}
    while i<len(A):
        j=0
        while j<len(Adomain[i]):

            if Adomain[i][j]==aval[i]:
                j+=1
                continue

            var_lst.append(m.add_var(var_type=BINARY))
            var_map[len(var_lst)-1]=(i,j)
            j+=1
        i+=1
    print ("beta list is ",betalst,beta0)
    cost_lst=[]
    constr_lst=[]
    constr_lst.append(beta0)
    iter=0
    i=0
    while i<len(A):
        j=0
        del_cons=[]
        while j<len(Adomain[i]):

            if Adomain[i][j]==aval[i]:
                constr_lst.append(betalst[i]*Adomain[i][j])
                j+=1
                continue
            #print (j,"ASDSA",Adomain[i][j],aval[i],Adomain[i][j]==aval[i])
            cost_lst.append(var_lst[iter] * (Adomain[i][j]-aval[i]))#Cost is value - original value
            constr_lst.append(betalst[i]*var_lst[iter]*(Adomain[i][j]-aval[i]))
            del_cons.append(var_lst[iter])
            iter+=1
            j+=1
        m += xsum(del_cons) <= 1
        i+=1

    i=len(A)
    while i<len(betalst):
        constr_lst.append(betalst[i]*kval[i-len(A)])
        i+=1
    #print (constr_lst)
    print ("Constraint threshold",math.log(alphak*1.0/(1-alphak)),alphak)
    m+=xsum(constr_lst)>=math.log(alphak*1.0/(1-alphak))
    m.objective = minimize(xsum(cost_lst))
    m.optimize()
    if m.num_solutions:
        print('Objective value %g found:'
                  % (m.objective_value))
        i=0
        while i<len(var_lst):
            print (i,var_lst[i].x,var_map[i])
            i+=1



def get_val(row,target,target_val):
    i=0
    while i<len(target):
        #print (row[target[i]],target_val[i])
        if not int(row[target[i]])==int(target_val[i]):
            return 0
        i+=1
    return 1

def get_prob_o_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))

    X=df[conditional]
    regr = LogisticRegression(random_state=0)#RandomForestRegressor(max_depth=5, random_state=0)
    regr.fit(X, new_lst)
    return (regr.predict_proba([conditional_values])[0][1])
    #return(regr.predict([conditional_values])[0])


def get_logistic_param(df,conditional,target,target_val):
    new_lst=[]
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))

    X=df[conditional]
    regr = LogisticRegression(random_state=0)
    regr.fit(X, new_lst)
    return (regr.coef_.tolist()[0],list(regr.intercept_))
