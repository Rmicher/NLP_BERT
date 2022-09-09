# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 00:06:24 2022

@author: rober
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import re
#import nltk


def make_charts(cols_, data_, name_):
    fig, axs = plt.subplots(nrows=int(np.ceil(cols_.shape[0]/3)), ncols=3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(name_, fontsize=18, y=0.95)
    
    # loop through tickers and axes
    for col, ax in zip(cols_, axs.ravel()):
        # filter df for ticker and plot on specified axes
        data_.loc[:, col].hist(ax=ax)
    
        # chart formatting
        ax.set_title(col)
        #ax.get_legend().remove()
        #ax.set_xlabel("")
    plt.savefig(name_ + " Histograms.png")
    plt.show()
    
    
test = pd.read_csv("test.csv")
train= pd.read_csv("train.csv")

sample_sub = pd.read_csv("sample_submission.csv")

sample_sub

train.max()
train.min()



set(train.loc[:,['answer_satisfaction']])

## Attribute Column Sorting ##
qtype_cols = train.iloc[:,11:].columns[train.iloc[:,11:].columns.str.match("^question_type")]
q_cols = train.iloc[:,11:].columns[ train.iloc[:,11:].columns.str.match("^question_") & ~train.iloc[:,11:].columns.str.match("^question_type")]
a_cols = train.iloc[:,11:].columns[train.iloc[:,11:].columns.str.match("^answer")]


dup_qtitle = train.loc[train.duplicated(subset=['question_title', 'question_body'], keep=False), :]



q_dup = dup_qtitle[['question_title', 'question_body'] + list(q_cols) + list(qtype_cols)]

qdup_summary = q_dup.groupby(['question_title', 'question_body']).agg(['max', 'min', 'mean', 'var', 'count'])

#qdup_summary.shape # (18358, 105)

#qdup_var = qdup_summary.loc[qdup_summary >0 ]

question_type_sum = train.loc[:, train.columns.str.match("^question_type")].sum(axis=1)
question_type_sum.hist()
question_type_sum.max()


op_fact = train.loc[:, ['question_fact_seeking', 'question_opinion_seeking']].sum(axis=1)
op_fact.hist()


### Making Comparable Histogragms ###
make_charts(cols_= a_cols, data_= train, name_= "Answer Attributes")
make_charts(cols_= q_cols, data_= train, name_= "Question Attributes")
make_charts(cols_= qtype_cols, data_= train, name_= "QType Attributes")


## Correlation Matrices
qtype_corr = train.loc[:, qtype_cols].corr()
a_corr = train.loc[:, a_cols].corr()
q_corr = train.loc[:, q_cols].corr()

all_corr = train.iloc[:,11:].corr()

# Interesting columns ['question_conversational', 'question_fact_seeking', 'question commonly accepted answer', 'question interestingess_others', 'not really a question']

#opinion seeking vs fact seeking - 68% fact seeking
# Question Type Instructions answer type instructions 
## Descriptors
col_summary = train.iloc[:,11:].describe()
col_nunique = train.iloc[:,11:].nunique()
qtype_counts = np.unique(train.loc[:, qtype_cols].values.ravel(), return_counts=True)






