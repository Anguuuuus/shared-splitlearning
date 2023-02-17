# import socket
import pickle

DAM = b'ok!'   # dammy 送信用

def recv_size_n_msg(s):
    exp_size = int(s.recv(16))
    s.sendall(DAM)  # send dammy
    recv_size = 0
    recv_data = b''
    while recv_size < exp_size:
        packet = s.recv(524288)
        # packet = s.recv(4096)
        recv_size = recv_size + len(packet)
        recv_data = recv_data + packet

    s.sendall(DAM)
    recv_data = pickle.loads(recv_data)

    return recv_data

def send_size_n_msg(msg, s):
    bytes = pickle.dumps(msg)
    msg_size = len(bytes)
    msg_size_bytes = str(format(msg_size, '16d')).encode()
    s.sendall(msg_size_bytes)   # send size
    dammy = s.recv(4)
    s.sendall(bytes)            # send msg
    dammy = s.recv(4)
