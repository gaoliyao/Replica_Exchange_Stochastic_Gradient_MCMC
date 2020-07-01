import os
import sys
import pickle

def write_chain(objects, num, folderName):
    if num < 0:
        return -1
    elif num == 0:
        # folder exists but empty: delete the folder
        if not os.path.isdir('./output/' + folderName): # and os.listdir(folderName == []):
            os.system('mkdir ./output/' + folderName)
        else:
            sys.exit('Folder ' + folderName + ' already existed!')
    with open('./output/' + folderName + '/' + str(num), "wb") as f:
        pickle.dump(objects, f, pickle.HIGHEST_PROTOCOL)

def read_chain(num, folderName):
    with open('./output/' + folderName + '/' + str(num), "rb") as f:
        myChain = pickle.load(f)
    return myChain

