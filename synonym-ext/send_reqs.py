import json

import grequests

url = 'http://localhost:3000/insert'
file_name = 'si_similar_beauti.json'
resultList = []
f = open(file_name)
all_dict = json.load(f)
f.close()
f = None
dict_keys_list = list(all_dict.keys())
for i in range(len(dict_keys_list)):
    comp = dict_keys_list[i]
    values = ""
    for j in all_dict[comp]:
        values += j + ","
    values = values[:len(values) - 1]
    jsn = dict()
    jsn["c"] = comp
    jsn["v"] = values
    action_item = grequests.post(url, json=jsn)
    resultList.append(action_item)
    if i % 1000 == 0:
        grequests.map(resultList)
        resultList = []

if len(resultList) != 0:
    grequests.map(resultList)
