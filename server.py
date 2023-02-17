''' 
Split learning (Client A -> Server -> Client A)
Server program
'''

from email.generator import BytesGenerator
import os
from pyexpat import model
import socket
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import sys
import copy
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from time import process_time

import MyNet
import socket_fun as sf
DAM = b'ok!'   # dammy 送信用

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("device: ", device)

mymodel = MyNet.MyNet_hidden().to(device)
print("mymodel: ", mymodel)

# -------------------- connection ----------------------
# connection establish
user_info = []
host = '0.0.0.0'
port = 19089
ADDR = (host, port)
s = socket.socket()
s.bind(ADDR)
USER = 1
s.listen(USER)
print("Waiting clients...")

# CONNECT
for num_user in range(USER):
    conn, addr = s.accept()
    user_info.append({"name": "Client "+str(num_user+1), "conn": conn, "addr": addr})
    print("Connected with Client "+str(num_user+1), addr)

# RECEIVE
for user in user_info:
    recvreq = user["conn"].recv(1024)
    print("receive request message from client <{}>".format(user["addr"]))
    user["conn"].sendall(DAM)   # send dammy

# ------------------- start training --------------------
def train(user):

    # store the time training starts
    p_start = process_time()

    i = 1
    ite_counter = -1
    user_counter = 0
    # PATH = []
    # PATH.append('./savemodels/client1.pth')
    # PATH.append('./savemodels/client2.pth')
    # PATH.append('./savemodels/client3.pth')
    lr = 0.005
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    LOADFLAG = 0

    while True:
        ### receive MODE
        recv_mode = sf.recv_size_n_msg(user["conn"])

        # ============= train mode ============
        if recv_mode == 0:
            mymodel.train()
            # if LOADFLAG == 1:
            #     mymodel.load_state_dict(torch.load(PATH[user_counter-1]))
            #     LOADFLAG = 0

            ite_counter+=1
            print("(USER {}) TRAIN Loading... {}".format(i, ite_counter))

            # RECEIVE ---------- feature data 1 ---------
            recv_data1 = sf.recv_size_n_msg(user["conn"])

            # 最適化関数リセット
            optimizer.zero_grad()

            # Forward prop. 2
            output_2 = mymodel(recv_data1)

            # SEND ------------ feature data 2 ----------
            sf.send_size_n_msg(output_2, user["conn"])

            # ==================== 順伝播終了 =====================

            # RECEIVE ------------ grad 2 ------------
            recv_grad = sf.recv_size_n_msg(user["conn"])

            # Back prop.
            output_2.backward(recv_grad)

            # update param.
            optimizer.step()

            # SEND ------------- grad 1 -----------
            sf.send_size_n_msg(recv_data1.grad, user["conn"])

        # ============= test mode =============
        elif recv_mode == 1:
            ite_counter = -1
            mymodel.eval()
            print("(USER {}) TEST Loading...".format(i))

            # RECEIVE ---------- feature data 1 -----------
            recv_data = sf.recv_size_n_msg(user["conn"])

            output_2 = mymodel(recv_data)

            # SEND ---------- feature data 2 ------------
            sf.send_size_n_msg(output_2, user["conn"])

        # =============== move to the next client =============
        elif recv_mode == 2:    # Epoch EACH verの場合
            ite_counter = -1
            # torch.save(mymodel.state_dict(), PATH[user_counter-1])
            # LOADFLAG = 1
            print(user["name"], " finished training!!!")
            i = i%USER
            print("現在user ", i+1, "です")
            user = user_info[i]
            i += 1
        
        # ============== this client done, move to the next client ==========
        elif recv_mode == 3:
            user_counter += 1
            i = i%USER
            # torch.save(mymodel.state_dict(), PATH[user_counter-1])
            # LOADFLAG = 1
            print(user["name"], "all done!!!!")
            user["conn"].close()
            if user_counter == USER: break
            user = user_info[i]
            i += 1

        else:   print("!!!!! MODE error !!!!!")

    print("=============Training is done!!!!!!===========")
    print("Finished the socket connection(SERVER)")

    # store the time training ends
    p_finish = process_time()

    print("Processing time: ",p_finish-p_start)

if __name__ == '__main__':
    train(user_info[0])