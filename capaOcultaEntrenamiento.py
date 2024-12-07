import cv2 as cv
import os
import numpy as np
from time import time
dataRuta='D:/Reconocimiento_Facial/Reconocimiento_facial_1/Data'
listaData=os.listdir(dataRuta)

ids=[]
rostrosData=[]
id=0
tiempoInicial=time()
for fila in listaData:
    rutaCompleta=dataRuta+'/'+fila
    print('inicio lectura')
    for archivo in os.listdir(rutaCompleta):

        print('Rostros:',fila+'/'+archivo)

        ids.append(id)
        rostrosData.append(cv.imread(rutaCompleta+'/'+archivo,0))


    id+=1
    tiempoFinal = time()
    tiempoTotal = tiempoFinal - tiempoInicial
    print('Tiempo total de lectura:', tiempoTotal)

entrenamientoModelo1=cv.face.EigenFaceRecognizer_create()
print('Entrenando...')
entrenamientoModelo1.train(rostrosData,np.array(ids))
TiempoFinalEntrenamiento = time()
tiempoTotalEntrenamiento = TiempoFinalEntrenamiento-tiempoTotal
print('Tiempo de entrenamiento:', tiempoTotalEntrenamiento)
entrenamientoModelo1.write('modeloEigenFace.xml')
print('Modelo entrenado')

