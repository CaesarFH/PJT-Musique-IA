import matplotlib.pyplot as plt
import numpy as np

#
# draw the look of commonly used activation functions
#
def heaviside(x):
    return (x>=0)*1

def sigmoide(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def atan(x):
    return np.arctg2(x)

def relu(x):
    return (x>=0)*x


listeF = [heaviside, sigmoide, tanh, relu]

plt.figure(figsize=(14,4))
plt.title("Exemples de fonction d'activation...")

plt.subplot(1,len(listeF),1)
x = np.linspace(-10,10,1000)
plt.plot(x, heaviside(x),'-r')
plt.title("Heaviside")
plt.xlim(-1,1)
locs, labels = plt.xticks()
plt.xticks(list(range(-3,4,1)),('','','','0','','',''))
plt.yticks(np.linspace(0,1,6),('0','','','','','1'))
plt.grid()

plt.subplot(1,len(listeF),2)
x = np.linspace(-7,8,100)
plt.plot(x, sigmoide(x),'-r')
plt.title("Sigmoïde")
plt.xlim(-6,6)
locs, labels = plt.xticks()
plt.xticks(list(range(-6,7,2)),('','','','0','','',''))
plt.yticks(np.linspace(0,1,6),('0','','','','','1'))
plt.grid()


plt.subplot(1,len(listeF),3)
x = np.linspace(-2,1,100)
plt.xlim(-2,1.1)
plt.plot(x, relu(x),'-r')
plt.title("'relu': rectified linear unit")
locs, labels = plt.xticks()
plt.xticks((-1.5,-1,-.5,0,.5,1),('','','','0','',''))
plt.yticks(np.linspace(0,1,6),('0','','','','','1'))
#plt.ylim(0,1)
plt.grid()

plt.subplot(1,len(listeF),4)
x = np.linspace(-6,6,100)
plt.plot(x, tanh(x),'-r')
plt.title("'tanh': tangente hyperbolique")
plt.xlim(-6,6)
locs, labels = plt.xticks()
plt.xticks(list(range(-6,7,2)),('','','','0','','',''))
plt.yticks(np.linspace(-1,1,7),('-1','','','0','','','1'))
plt.grid()

plt.savefig("activationFunctions.png")
plt.subplots_adjust(wspace=0.3, top=0.9, bottom=0.09, left=.02, right=.99)
plt.show()
