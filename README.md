# IA-tp-2

## Instalar
- python 3.3.6
- Si usás Ubuntu, puede ser que necesites instalar el Tkinter con sudo apt-get install python3.6-tk
- pip3 install -r requirements.txt

## Entrenamiento
> Lanzar el entrenamiento `python3 train.py`
Se pueden modificar las constantes CONST_LEARN_RATE y CONST_EPOCHS para cambiar el factor de aprendizaje y los epochs para cambiar la cantidad de ciclos de entrenamiento

## Predecir valores
Una vez que tenemos el modelo armado, podemos correr la predicción para un usuario particular CONST_USER_INDEX 
> Lanzar el predict `python3 predict.py`
Esto devolverá la predicción para el usuario del CONST_USER_INDEX (índice del usuario en el array de testeo) y otro para un usuario generado de forma aleatoria