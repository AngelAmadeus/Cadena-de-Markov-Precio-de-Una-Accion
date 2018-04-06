#Practica 10: Datareader, Series de Tiempo y Markov 
import numpy as np
import pandas as pd
#from datetime import datetime
import matplotlib.pyplot as plt
#import pandas_datareader.data as web
#from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hc
#%%Importación de datos
#start = datetime(2017,2,28)
#end = datetime(2018,2,28)
#data = web.DataReader('GrumaB.MX','yahoo',start,end)
#%%Improtación de datos 2
data_file = '..\Data\GRUMAB.MX.csv'
data = pd.read_csv(data_file, header = 0)
#%%Visualizar los Datos
plt.plot(data['Close'],'b-')#linea azul y continua
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid()
plt.show()
#Se van a generar pequeñas series de tiempo de la serie de tiemplo completa
#De esas mini series de tiempo que son como pequeñas ventanas se analizara el patrón
#Y se va a comparar entre ventanas. Si tomamos ventanas de 5 días, una ventana sería
#e 1-5, 2-6, 3-7 y así sucesivamente
#Si se observa de forma matricial, decir que
#[1,2,3,4,5;2,3,4,5,6;3,4,5,6,7;....;248,249,250,251,251]
#Esto sería hacer una ciclo de 248 o en su caso solo de 5 ciclos por el acomodo de
#la matriz para hacerlo más rápido
dat = data['Close']
num_ventana = 5
nprices = len(dat)
dat_new = np.zeros((nprices-num_ventana,num_ventana))
for k in np.arange(num_ventana):
    dat_new[:,k]=dat[k:nprices-num_ventana+k]
#%%Visualizar la nueva serie de tiempo
plt.plot(dat_new.transpose())#miniseries de tiempo
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid()
plt.show()
#%%
tmp = dat_new.transpose()
dat_new = np.transpose((tmp-tmp.mean(axis=0))/tmp.std(axis=0))
del tmp
plt.plot(dat_new.transpose())#miniseries de tiempo
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid()
plt.show()
#%%Aplicar el algoritmo de hierarchical clustering
Z = hc.linkage(dat_new,'ward')
plt.figure()
hc.dendrogram(Z)
plt.show()
#%%Generar grupos con axima distancia de 10
max_d = 10 #esto es lo que me hace los grupos
clusters = hc.fcluster(Z,max_d,criterion='distance')
np.unique(clusters)
#%%Visualizar Cluster 1 (se pueden visualizar todos)
cluster = 1
indx = clusters == cluster
datclust = dat_new[indx,]
media_datclust = datclust.mean(axis=0)
plt.subplot(211)
plt.plot(datclust.transpose())
plt.xlabel('Time')
plt.ylabel('Price')
plt.subplot(212)
plt.plot(media_datclust.transpose())
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
#%%Ver todas las tendencias
n_subfig = np.ceil(np.sqrt(len(np.unique(clusters))))
for k in np.arange(1,len(np.unique(clusters))+1):
    datclust = dat_new[clusters==k]
    plt.subplot(n_subfig,n_subfig,k)
    plt.plot(datclust.mean(axis=0))
    plt.ylabel('cluster %d'%k)
plt.show()
#%%Interpretación
#esto me dice que "las ventanitas" pertenecen a cual cluster
#Podrían considerarse como señales de trading
plt.subplot(211)
plt.plot(dat)
plt.title('Serie de Tiempo')
plt.xlabel('Time')
plt.ylabel('Price')
plt.subplot(212)
plt.bar(np.arange(num_ventana,len(dat)),clusters)
plt.xlabel('Time')
plt.ylabel('Group')
plt.show()
#Visualizar un patron o grupo en la serie de tiempo
num_cluster_color = 3
pos = np.arange(num_ventana,len(dat))[clusters==num_cluster_color]
#de pos tomo los 5 días anteriores y los pinto de otro color
plt.figure(figsize=(8,8))
plt.plot(np.array(dat),'b-')
for k in pos:
    plt.plot(np.arange(k-num_ventana+1,k),np.array(dat[k-num_ventana+1:k]),'r-')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
#Esto no nos sirve muecho en finanzas: por ejemplo si se toma el comportamiento alsista,
#el algoritmo te dice que es alsista, sin embargo, te lo dice una vex que estás arriba,
#sería deseable que te lo dijera antes de que legaras a ese punto. Por lo anterior se 
#utilizan Cadenas de Markov para inferir de acuerdo a probabilidades, es decir,
#la probabilidad de que pasa algo, sabiendo que estoy en un punto particular,
#y así con todas las combinaciones
#%%Cadenas de Markov
num_clusters = np.unique(clusters)
M = np.zeros((len(num_clusters),len(num_clusters)))
num_patrones = len(clusters)
secuencias_par_clusters = np.zeros((num_patrones-1,2))
#esto es para ver que número sigue de cual. Para simplificar el acomodo solo se copia
#la columna junto con si misma, pero quitando el primer dato, para ver la secuencia,
#es decir que:
secuencias_par_clusters[:,0] = clusters[0:num_patrones-1]
secuencias_par_clusters[:,1] = clusters[1:num_patrones]
#a continuación se hace la matriz de cunantas veces estando en el patron 1
#pasé al patron 2, 3 4 y 5 y así sucesivamente el 2 con el 1,3,4,5
for k in num_clusters:
    for l in num_clusters:
        indx =(secuencias_par_clusters[:,0]==k)&(secuencias_par_clusters[:,1]==l)
        M[l-1,k-1] = np.sum(indx)
M=M/M.sum(axis=0)
#%%Simulación de los patrones siguientes ( Prob(X!) = Prob(X1 dado que X0)Prob(X=) )
#Dentro de lo demás es ( X1 = M X0)
X0 = np.zeros((len(num_clusters),1))
X0[0,0] = 1 #esto es sabiendo que y estoy en el cluster 1
#X0[1,0] = 1 #esto es sabiendo que y estoy en el cluster 2
#X0[2,0] = 1 #esto es sabiendo que y estoy en el cluster 3
#X0[3,0] = 1 #esto es sabiendo que y estoy en el cluster 4
#X0[4,0] = 1 #esto es sabiendo que y estoy en el cluster 5
X1 = np.matrix(M)*np.matrix(X0)
X2 = np.matrix(M)*np.matrix(X1)



#logica difusa == inventarte reglas y umbrales de acuerdo a ciertas cosas
#ejemplo de temperatura 