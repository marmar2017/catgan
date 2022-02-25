import pandas as pd
import csv
from os import path
import random

def divide_list(data, percent=0.9):
    length = len(data)
    return data[0:int(length*percent)], data[int(length*percent):-1]


def write_to_file(data, f):
    for d in data:
        print(d, file=f)

dataset_path = '/home/user1/Ru_experiement/TextGAN-PyTorch/dataset'
def Coursera_Reviews_handling():
    df = pd.read_csv(path.join(dataset_path,'Coursera_Reviews.txt'),delimiter='	',header=None)
    pos, neu, neg = 0,0,0
    total_final = 0
    total = len(df)
    review_len=150
    dataset_name = 'cr%s'%str(review_len)
    sr = open(path.join(dataset_path, '%s.txt'%dataset_name), mode='w', encoding='utf-8')
    sr_0 = open(path.join(dataset_path, '%s_cat0.txt'%dataset_name),
                mode='w', encoding='utf-8')  # pos
    sr_1 = open(path.join(dataset_path, '%s_cat1.txt'%dataset_name),
                mode='w', encoding='utf-8')  # neu
    sr_2 = open(path.join(dataset_path, '%s_cat2.txt'%dataset_name),
                mode='w', encoding='utf-8')  # neg

    sr_test = open(path.join(dataset_path, 'testdata/%s_test.txt'%dataset_name),
                mode='w', encoding='utf-8')
    sr_0_test = open(path.join(dataset_path, 'testdata/%s_cat0_test.txt'%dataset_name),
                    mode='w', encoding='utf-8')  # pos
    sr_1_test = open(path.join(dataset_path, 'testdata/%s_cat1_test.txt'%dataset_name),
                    mode='w', encoding='utf-8')  # neu
    sr_2_test = open(path.join(dataset_path, 'testdata/%s_cat2_test.txt'%dataset_name),
                    mode='w', encoding='utf-8')  # neg


    r_t = []
    r_p = []
    r_n = []
    r_neu = []

    for i in range(len(df)):
        review, label = df[2][i], df[1][i]
        # using lower and upper method to judge whether it is alphabet
        if len(review) < review_len:
            total_final += 1
            r_t.append(review)
            if label == 'P':
                pos += 1
                r_p.append(review)
            elif label == 'NEU':
                neu += 1
                r_neu.append(review)
            else:
                neg += 1
                r_n.append(review)
        if i % 1000 == 0:
            print('have handeled %d reviews...' % i)

    print('there are total %d reviews. There are %d reviews whose length is not bigger than %d' %
        (total, total_final, review_len))
    print('Whole dataset, there are total %d reviews containing %d postive reviews, %d negative reviews, %d neutral reviews in total' %
        (len(r_t), len(r_p), len(r_n), len(r_neu)))


    random.shuffle(r_p)
    random.shuffle(r_n)
    random.shuffle(r_neu)


    # divide dataset to train and test 90%
    r_p, r_p_test = divide_list(r_p)
    r_n, r_n_test = divide_list(r_n)
    r_neu, r_neu_test = divide_list(r_neu)
    r_t, r_t_test = r_p + r_n + r_neu, r_p_test+r_n_test+r_neu_test

    write_to_file(r_t, sr)
    write_to_file(r_p, sr_0)
    write_to_file(r_neu, sr_1)
    write_to_file(r_n, sr_2)

    write_to_file(r_t_test, sr_test)
    write_to_file(r_p_test, sr_0_test)
    write_to_file(r_neu_test, sr_1_test)
    write_to_file(r_n_test, sr_2_test)

    print('Train dataset, there are total %d reviews containing %d postive reviews, %d negative reviews, %d neutral reviews in total' %
        (len(r_t), len(r_p), len(r_n), len(r_neu)))
    print('Test dataset, there are total %d reviews containing %d postive reviews, %d negative reviews, %d neutral reviews in total' %
        (len(r_t_test), len(r_p_test), len(r_n_test), len(r_neu_test)))

    sr.close()
    sr_0.close()
    sr_1.close()
    sr_2.close()
    sr_test.close()
    sr_0_test.close()
    sr_1_test.close()
    sr_2_test.close()




# get_reviews_length_amazon_book()
Coursera_Reviews_handling()