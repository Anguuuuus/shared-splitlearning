from matplotlib import pyplot as plt
import csv
''' 
csv format is like below, (without labels)
------------------------------------------------------------------
client No,  train_loss,  train_acc,  val_loss,  val_acc,  p_time
------------------------------------------------------------------
client (without sharing),    [],          [],         [],        [],       []
client1,                     [],          [],         [],        [],       []
client2,                     [],          [],         [],        [],       []
client3,                     [],          [],         [],        [],       []
'''

train_loss, train_acc, val_loss, val_acc = [], [], [], []
p_time = []
cal_times = []
USER = 4
epochs = 50

file = './results/csv/model0206.csv'

def make_list_from_csv(file, USER):
    with open(file, newline='') as f:
        counter = 0
        csvreader = csv.reader(f)
        content = [row for row in csvreader]    # [ [row],[row],[row],[row] ]
        for list in content:
            train_loss.append(list[1])
            train_acc.append(list[2])
            val_loss.append(list[3])
            val_acc.append(list[4])
            p_time.append(list[5])
            counter+=1
            if counter == USER: break

make_list_from_csv(file, USER)      # val_loss[0~2], val_acc[0~2]

for i in range(USER):
    val_acc[i] = eval(val_acc[i])
    val_acc[i].insert(0, 0.0)

epoch_over_eighty = []
time_over_eighty = []
for user in p_time:
    tmp = 0.0
    user = eval(user)
    cal_time = []
    cal_time.append(0.0)
    for time in user:
        tmp+=float(time)
        cal_time.append(tmp)
    cal_times.append(cal_time)


# 80%を超えたtimeを格納
for i in range(USER):
    # val_acc[i] = eval(val_acc[i])
    for j in range(len(val_acc[i])):
        if val_acc[i][j]*100 > 80:
            epoch_over_eighty.append(j)
            # time_over_eighty.append(cal_times[i][j])
            print("client", i, cal_times[i][j])
            break
            
# -------------------------------------------------------------
# print("client1 (cond.I) processing time: ", cal_times[0][50])
# print("client2 (cond.I) processing time: ", cal_times[1][50])
# print("client3 (cond.I) processing time: ", cal_times[2][50])
# print("client1 (cond.II)processing time: ", cal_times[3][50])
# print("client2 (cond.II)processing time: ", cal_times[4][50])
# print("client3 (cond.II)processing time: ", cal_times[5][50])

print("client (without sharing) processing time: ", cal_times[0][50])
print("client1 processing time: ", cal_times[1][50])
print("client2 processing time: ", cal_times[2][50])
print("client3 processing time: ", cal_times[3][50])
# -------------------------------------------------------------

# ==============================================================
# print("Max acc of client 1 (cond.I): ", max(val_acc[0]))
# print("Max acc of client 2 (cond.I): ", max(val_acc[1]))
# print("Max acc of client 3 (cond.I): ", max(val_acc[2]))
# print("Max acc of client 1 (cond.II): ", max(val_acc[3]))
# print("Max acc of client 2 (cond.II): ", max(val_acc[4]))
# print("Max acc of client 3 (cond.II): ", max(val_acc[5]))

print("Max acc of client (without sharing): ", max(val_acc[0]))
print("Max acc of client 1: ", max(val_acc[1]))
print("Max acc of client 2: ", max(val_acc[2]))
print("Max acc of client 3: ", max(val_acc[3]))
# ===============================================================

# print("client1 epoch(80>): ", epoch_over_eighty[0])
# print("client2 epoch(80>): ", epoch_over_eighty[1])
# print("client3 epoch(80>): ", epoch_over_eighty[2])
# print("client(ws) epoch(80>): ", epoch_over_eighty[3])


# original colors -> black, blue, green, red

plt.figure()
# plt.plot(cal_times[0], val_acc[0], color='blue',linewidth=1, linestyle='--', label='Client 1 in Cond. I')
# plt.plot(cal_times[1], val_acc[1], color='green',linewidth=1, linestyle='--', label='Client 2 in Cond. I')
# plt.plot(cal_times[2], val_acc[2], color='red',linewidth=1, linestyle='--', label='Client 3 in Cond. I')
# plt.plot(cal_times[3], val_acc[3], color='blue',linewidth=1, linestyle='-', label='Client 1 in Cond. II')
# plt.plot(cal_times[4], val_acc[4], color='green',linewidth=1, linestyle='-', label='Client 2 in Cond. II')
# plt.plot(cal_times[5], val_acc[5], color='red',linewidth=1, linestyle='-', label='Client 3 in Cond. II')

plt.plot(cal_times[0], val_acc[0], color='black',linewidth=1, linestyle='--', label='Client (without sharing)')
plt.plot(cal_times[1], val_acc[1], color='blue',linewidth=1, linestyle='-', label='Client 1 in this simulation')
plt.plot(cal_times[2], val_acc[2], color='green',linewidth=1, linestyle='-', label='Client 2 in this simulation')
plt.plot(cal_times[3], val_acc[3], color='red',linewidth=1, linestyle='-', label='Client 3 in this simulation')

plt.legend(fontsize=14)
plt.xlabel('Proccesing time [s]', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tick_params(labelsize=14)
plt.xlim(0,)
plt.ylim(0,)
plt.grid()
plt.subplots_adjust(left=0.140, right=0.980, bottom=0.130, top=0.970)
plt.savefig('./results/acc/model0207/time-2.png')


plt.figure()
# plt.plot(range(epochs+1), val_acc[0], color='blue',linewidth=1, linestyle='--', label='Client 1 in Cond. I')
# plt.plot(range(epochs+1), val_acc[1], color='green',linewidth=1, linestyle='--', label='Client 2 in Cond. I')
# plt.plot(range(epochs+1), val_acc[2], color='red',linewidth=1, linestyle='--', label='Client 3 in Cond. I')
# plt.plot(range(epochs+1), val_acc[3], color='blue',linewidth=1, linestyle='-', label='Client 1 in Cond. II')
# plt.plot(range(epochs+1), val_acc[4], color='green',linewidth=1, linestyle='-', label='Client 2 in Cond. II')
# plt.plot(range(epochs+1), val_acc[5], color='red',linewidth=1, linestyle='-', label='Client 3 in Cond. II')

plt.plot(range(epochs+1), val_acc[0], color='black',linewidth=1, linestyle='--', label='Client (without sharing)')
plt.plot(range(epochs+1), val_acc[1], color='blue',linewidth=1, linestyle='-', label='Client 1 in this simulation')
plt.plot(range(epochs+1), val_acc[2], color='green',linewidth=1, linestyle='-', label='Client 2 in this simulation')
plt.plot(range(epochs+1), val_acc[3], color='red',linewidth=1, linestyle='-', label='Client 3 in this simulation')

plt.legend(fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tick_params(labelsize=14)
plt.xlim(0,)
plt.ylim(0,)
plt.grid()
plt.subplots_adjust(left=0.140, right=0.980, bottom=0.130, top=0.970)
plt.savefig('./results/acc/model0207/epoch-2.png')