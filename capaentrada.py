import cv2 as cv
import os
import imutils

modelo='FotosArturo'
ruta1='D:/Reconocimiento_Facial/Reconocimiento_facial_1/Data'
rutacomplete=ruta1+'/'+modelo
if not os.path.exists(rutacomplete):
    os.makedirs(rutacomplete)
    print('Carpeta creada:',rutacomplete)



camara = cv.VideoCapture(0)
ruidos=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
id=350
while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    captura=imutils.resize(captura,width=640)

    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura=captura.copy()

    cara=ruidos.detectMultiScale(grises,1.3,5)

    for(x,y,w,h) in cara:
        cv.rectangle(captura,(x,y),(x+w,y+h),(0,255,0),2)
        rostrocapturado=idcaptura[y:y+h,x:x+w]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacomplete+'/imagen_{}.jpg'.format(id),rostrocapturado)
        id+=1

    cv.imshow("Resultado rostro",captura)

    if id==500:
        break
camara.release()
cv.destroyAllWindows()
