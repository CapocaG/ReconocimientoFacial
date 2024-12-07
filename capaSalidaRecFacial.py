import cv2 as cv
import os
import imutils
import pywhatkit as kit
import pyautogui
import time


dataRuta = 'D:/Reconocimiento_Facial/Reconocimiento_facial_1/Data'
listaData = os.listdir(dataRuta)
modeloEigenFace = cv.face.EigenFaceRecognizer_create()
modeloEigenFace.read('modeloEigenFace.xml')
ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
camara = cv.VideoCapture(0)

while True:
    respuesta, captura = camara.read()
    if not respuesta:
        break
    captura = imutils.resize(captura, width=640)
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura = grises.copy()
    caras = ruidos.detectMultiScale(grises, 1.3, 5)



    for (x, y, w, h) in caras:
        rostrocapturado = idcaptura[y:y + h, x:x + w]
        rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = modeloEigenFace.predict(rostrocapturado)
        cv.putText(captura, '{}'.format(resultado), (x, y - 5), 1, 1.3, (255, 0, 0), 1, cv.LINE_AA)
        if resultado[1] < 8500:  # Ajusta este umbral según sea necesario
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x, y - 20), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv.putText(captura, 'Desconocido', (x, y - 20), 2, 0.8, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Enviar mensaje a WhatsApp
            try:
                kit.sendwhatmsg_instantly("+51921786923", "Se ha detectado un intruzo por alrededor de la casa.")
                time.sleep(3)  # Esperar a que se abra WhatsApp Web y se escriba el mensaje
                pyautogui.press('enter')  # Simular la pulsación de la tecla Enter

                print('Mensaje enviado')
            except Exception as e:
                print('Error al enviar el mensaje:', e)
    cv.imshow('Resultado', captura)
    if cv.waitKey(1) == ord('s'):
        break

camara.release()
cv.destroyAllWindows()