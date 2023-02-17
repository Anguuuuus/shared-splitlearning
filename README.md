# shared-splitlearning
This repository contains a method to share intermediate layers in split learning.


---------- How to run the shared-SL program ----------
1. python server_save.py(or server.py)
2. python client1.py
3. python clientX.py

You can choose the number of clients which participate in the network.
In this repository, three clients are introduced.
If you want to increase the clients, duplicate the client's program.



-------- About server program ----------

server_save.py does saving and loading trained models just in case.
If you prefer without that processes, run server.py file.
