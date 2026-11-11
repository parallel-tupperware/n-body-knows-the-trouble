import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import math

mG = 1

N = 3
T_Steps = 30
dt = 0.1

Side_Len = 10

Pos = np.zeros((2, N, T_Steps + 1))
Vel = np.zeros((2, N, T_Steps + 1))
Current_Acc = np.zeros((2,N))

Pos_0 = Side_Len * np.random.rand(2, N)
Pos[:,:,0] = Pos_0

fig, ax = plt.subplots()
ax.set_xlim(-1, 15) #set here the view limits of the x-axis
ax.set_ylim(-1, 15) #same only for y-axis


def Velocity_Verlet(Current_Pos, Current_Vel, Current_Acc):
    Predicted_Pos = Current_Pos + Current_Vel * dt + 0.5 * Current_Acc * dt ** 2
    Predicted_Acc = Calc_acc(Predicted_Pos)
    Predicted_Vel = Current_Vel + 0.5 * dt * (Current_Acc + Predicted_Acc)
    return Predicted_Pos, Predicted_Vel, Predicted_Acc

def Step(i):
    global Current_Acc
    Pos[:,:,i + 1], Vel[:,:,i + 1], Current_Acc = Velocity_Verlet(Pos[:,:,i], Vel[:,:,i], Current_Acc)
    

def Loop():
    global Current_Acc
    for i in range(T_Steps): #to prevent overflow
        if(i == 0):
            Current_Acc = Calc_acc(Pos[:,:,i])
        Step(i)
    


def Calc_dist(Current_Pos):
    dist = Current_Pos.reshape(2, N, 1) - Current_Pos.reshape(2, 1, N)
    r = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
    
    norm_dist = np.nan_to_num(dist / r)
    return norm_dist, r

def Calc_acc(Current_Pos):
    norm_dist, r = Calc_dist(Current_Pos)
    acc = np.sum(np.nan_to_num(norm_dist * mG / (r ** 2)), axis=1) #since all masses are equal the m^2 becomes m
    return acc


Loop()


#make initial scatter plot
Scatter = ax.scatter(Pos[0,:,0], Pos[1,:,0], s = 10)

ax.scatter(Pos[0,:,0], Pos[1,:,0], s = 10)
ax.scatter(Pos[0,:,-1], Pos[1,:,-1], s = 10)
plt.show()


#function to extract the new positions
#def update(frame):
#    scatter.set_offsets(Pos[:,:,frame])
#    return scatter

#create animation
#Writer = animation.FFMpegWriter(fps = 30, bitrate = 1e4)
#ani = animation.FuncAnimation(fig, update, frames = T_Steps, interval = dt * 1000)
#ani.save("C:/Users/mathe/Documents/TU delft/AP/Y1/Semester 1/HPC/N-body/Video.mp4", writer = Writer)
#plt.show()
#print("Done!")
