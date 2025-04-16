

import os

def writeToReport(path, content):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "a") as f:
        f.write(content + '\n')


def list_to_str(list):
    str_list = ''
    for l in list:
        #print(l)
        str_list += str(l) + ','
    return str_list