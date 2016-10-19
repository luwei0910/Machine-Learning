# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:17:46 2016

@author: wei.lu
"""

import pandas
import csv

user_col = pandas.read_csv('C:/Users/wei.lu/Desktop/fresh_comp_offline/tianchi_fresh_comp_train_user_1.csv')
item_col = pandas.read_csv('C:/Users/wei.lu/Desktop/fresh_comp_offline/tianchi_fresh_comp_train_item.csv')

user_unique = user_col["user_id"].unique()
print user_unique
#user_unique = [10001082]

item_buy = {}
item_check = {}
item_favor = {}
item_cart = {}
for user in user_unique:
  item_check_user = user_col[(user_col["user_id"] == user) & (user_col["behavior_type"] == 1)]["item_id"]
  item_check_user_unique = item_check_user.unique()
  item_favor_user = user_col[(user_col["user_id"] == user) & (user_col["behavior_type"] == 2)]["item_id"]
  item_cart_user = user_col[(user_col["user_id"] == user) & (user_col["behavior_type"] == 3)]["item_id"]
  item_buy_user = user_col[(user_col["user_id"] == user) & (user_col["behavior_type"] == 4)]["item_id"]
  item_buy_user_unique = item_buy_user.unique()
  item_check[user] = item_check_user_unique
  item_favor[user] = item_favor_user
  item_cart[user] = item_cart_user
  item_buy[user] = item_buy_user_unique

submission = file('C:/Users/wei.lu/Desktop/fresh_comp_offline/tianchi_mobile_recommendation_predict.csv','wb')
writer = csv.writer(submission)
writer.writerow(['user_id','item_id'])

for user in user_unique:
  #print user,len(item_check[user]),len(item_favor[user]),len(item_cart[user]),len(item_buy[user])
  for item in item_buy[user]:
    writer.writerow([user,item])

submission.close()