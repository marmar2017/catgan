import pandas as pd
import csv
from os import path
dataset_path = '/home/user1/Ru_experiement/TextGAN-PyTorch/dataset'
def Coursera_Reviews_handling():
    df = pd.read_csv(path.join(dataset_path,'Coursera_Reviews.txt'),delimiter='	',header=None)
    pos, neu, neg = 0,0,0
    total = len(df)
    sr=open(path.join(dataset_path,'cr.txt'), mode='w',encoding='utf-8')
    sr_0 = open(path.join(dataset_path,'cr_0.txt'), mode='w',encoding='utf-8') #pos
    sr_1 = open(path.join(dataset_path,'cr_1.txt'), mode='w',encoding='utf-8') #neu
    sr_2 = open(path.join(dataset_path,'cr_2.txt'), mode='w',encoding='utf-8') #neg
    review_length = []
    for i in range(len(df)):
        review, label = df[2][i], df[1][i]
        # using lower and upper method to judge whether it is alphabet
        review_length.append(len(review))
        review +='\n'
        sr.write(review)
        if label == 'P':
            pos+=1
            sr_0.write(review)
        elif  label == 'NEU':
            neu+=1
            sr_1.write(review)
        else:
            neg+=1
            sr_2.write(review)
        if i %100 ==0:
            print('have handeled %d reviews...'%i)
    review_length.sort(reverse = True)
    review_wr=open('review_length.txt', mode='w',encoding='utf-8')
    for l in review_length:
        print(l,file = review_wr)
    print('there are total %d reviews, which include %d postive reviews, %d negative reviews, %d neutral reviews'%(total,pos,neg,neu))

    sr.close()
    sr_0.close()
    sr_1.close()
    sr_2.close()
    review_wr.close()

def get_reviews_length_amazon_book():
    df = pd.read_csv(path.join(dataset_path,'amazon_app_book.txt'),delimiter='	',header=None)
    review_length = []
    for i in range(len(df)):
        review_length.append(len(df[0][i]))
    review_length.sort(reverse = True)
    review_wr=open('amazon_book_review_length.txt', mode='w',encoding='utf-8')
    for l in review_length:
        print(l,file = review_wr)
    review_wr.close()

def get_reviews_length_sr():
    df = pd.read_csv(path.join(dataset_path,'sr.txt'),delimiter='	',header=None)
    review_length = []
    for i in range(len(df)):
        review_length.append(len(df[0][i]))
    review_length.sort(reverse = True)
    review_wr=open('sr_review_length.txt', mode='w',encoding='utf-8')
    for l in review_length:
        print(l,file = review_wr)
    review_wr.close()


# get_reviews_length_amazon_book()
get_reviews_length_sr()