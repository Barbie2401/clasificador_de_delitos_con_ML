# FUNCIONES AUXILIARES

import datetime
import re
from nltk.corpus import stopwords
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

VOCALES_TILDES = 'áéíóúü'
VOCALES_SIN_TILDES = 'aeiouu'
MAKE_TRANS = str.maketrans(VOCALES_TILDES,VOCALES_SIN_TILDES)


def transformar_a_fecha(fecha):
    """Transforma una fecha en varios formatos a formato datetime
    Input:
        fecha: str, fecha en formato dd/mm/aaaa, dd-mm-aaaa, dd.mm.aaaa, dd/mm/aa, dd-mm-aa, dd.mm.aa
    Output:
        fecha: datetime, fecha en formato datetime"""
    if isinstance(fecha, datetime.datetime):
        return fecha
    elif isinstance(fecha, datetime.date):
        return datetime.datetime.combine(fecha, datetime.time())
    # largo 10
    elif isinstance(fecha, str) and len(fecha) == 10 and fecha.count('-') == 2 and fecha[4] == '-':
        return datetime.datetime.strptime(fecha, "%Y-%m-%d")
    elif isinstance(fecha, str) and len(fecha) == 10 and fecha.count('-') == 2 and fecha[2] == '-':
        return datetime.datetime.strptime(fecha, "%d-%m-%Y")
    elif isinstance(fecha, str) and len(fecha) == 10 and fecha.count('/') == 2 and fecha[4] == '/':
        return datetime.datetime.strptime(fecha, "%Y/%m/%d")
    elif isinstance(fecha, str) and len(fecha) == 10 and fecha.count('/') == 2 and fecha[2] == '/':
        return datetime.datetime.strptime(fecha, "%d/%m/%Y")
    # largo 8
    elif isinstance(fecha, str) and len(fecha) == 8 and fecha.count('-') == 2 and fecha[4] == '-':
        return datetime.datetime.strptime(fecha, "%Y-%m-%d")
    elif isinstance(fecha, str) and len(fecha) == 8 and fecha.count('-') == 2 and fecha[2] == '-':
        return datetime.datetime.strptime(fecha, "%d-%m-%Y")
    elif isinstance(fecha, str) and len(fecha) == 8 and fecha.count('/') == 2 and fecha[4] == '/':
        return datetime.datetime.strptime(fecha, "%Y/%m/%d")
    elif isinstance(fecha, str) and len(fecha) == 8 and fecha.count('/') == 2 and fecha[2] == '/':
        return datetime.datetime.strptime(fecha, "%d/%m/%Y")
    else:
        return None

def contar_frecuencia(lista):
    """Cuenta la frecuencia de los elementos de una lista
    Input:
        lista: list, lista de elementos
    Output:
        frecuencia: dict, diccionario con los elementos de la lista como llaves y la frecuencia como valor"""
    frecuencia = {}
    for elemento in lista:
        if elemento in frecuencia:
            frecuencia[elemento] += 1
        else:
            frecuencia[elemento] = 1
    return frecuencia

def extraer_mas_frecuentes(diccionario, cantidad_de_elementos, mayor=True):
    """Extrae los elementos más frecuentes de un diccionario
    Input:
        diccionario: dict, diccionario con los elementos de la lista como llaves y la frecuencia como valor
        cantidad_de_elementos: int, cantidad de elementos a extraer
        mayor: bool, si es True extrae los elementos más frecuentes, si es False extrae los elementos menos frecuentes
    Output:
        mas_frecuentes: dict, lista con los elementos más frecuentes"""
    mas_frecuentes = {}
    if mayor:
        for i in range(cantidad_de_elementos):
            if len(diccionario) == 0:
                break
            elemento = max(diccionario, key=diccionario.get)
            mas_frecuentes[elemento] = diccionario[elemento]
            del diccionario[elemento]
    else:
        for i in range(cantidad_de_elementos):
            if len(diccionario) == 0:
                break
            elemento = min(diccionario, key=diccionario.get)
            mas_frecuentes[elemento] = diccionario[elemento]
            del diccionario[elemento]
    return mas_frecuentes

def leer_cvs_to_list(archivo, header=True):
    """Lee archivos de texto plano de una sola columna, los procesa para quitar tildes y espacios,
    los deja en minuscula y los retorna como lista
    Input:
        archivo: str, ruta del archivo de texto
        header: bool, define si el archivo trae nombre de la columna o no
    Output:
        lista: list, lista con las palabras encontradas en el archivo
    """
    i = 1 if header else 0
    with open(archivo, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lista = [x[0].lower().translate(MAKE_TRANS).strip() for x in reader]
        return lista[i:]

def text_preprocessor(text, stopwords):
    """Proceso para normalizar los textos con las siguientes acciones:
    - transforma en minuscula y quita espacios al inicio y termino
    - quita codigos de denuncuas "w202%"
    - quita tildes de vacales
    - deja solo vocales y espacios
    - quita espacios multiples
    - quita palabras encontradas en el stopwords
    Input:
        text: str, texto a procesar
        stopwords: list, listado de palabras a eliminar del texto
    Output:
        text: text, palabra procesada
    """
    text = text.lower().strip()
    text = re.sub(r'(w202)+\d{14}.', '',text).strip()
    text = text.translate(MAKE_TRANS)
    text = "".join([x for x in text if x.isalpha() or x == " "])
    text = " ".join(re.split(r"\s+", str(text)))
    text = re.sub(r'[^\w\s]', '', str(text)).strip()
    text = [w for w in text.split() if w not in stopwords]
    text = ' '.join(text)
    return text

def predict_category(model, vectorizer, label_encoder, listado_textos):
    '''Predice las categorias de un listado de textos
    Input:
        model: objeto del modelo a evaluar
        vectorizer: objeto con la vectorización utilizada
        label_encoder: objeto con la codificacion de etiquetas
        listado_textos: lista con los textos a clasificar
    Output:
        y_hat: list, clases codificadas
        y_hat_labels: list, clases decodificadas
        y_proba: list, listado con el listado de las probabilidades de clases para cada glosa
    '''
    df_listado_textos = pd.DataFrame(listado_textos, columns=['texto'])
    X = vectorizer.transform(df_listado_textos['texto'])
    y_hat = model.predict(X)
    y_hat_labels = [label_encoder.classes_[int(i)] for i in y_hat]
    y_proba = model.predict_proba(X)
    
    return y_hat,y_hat_labels,y_proba



#Observamos las diferentes columnas que hay entre los DataFrame:

def compara_columns_df (df_1,df_2):
    """
    Compara las columnas entre dos Data Frames
	    Parametros:
		    df_1: DataFrame 1
            df_2: DataFrame 2
	
        Retorno:
		    Entrega los campos que difieren entre un DataFrame y el otro. 
	"""	
    #Diferencia entre DF 1 y DF 2:
    diferencia_1 = set(df_1.columns.values).difference(set(df_2.columns.values))

    #Diferencia entre DF 2 y DF 1:
    diferencia_2 = set(df_2.columns.values).difference(set(df_1.columns.values))

    print(f'Campos que difieren entre DataFrame 1 y DataFrame 2: \n {diferencia_1}')

    print(f'\nCampos que difieren entre DataFrame 2 y DataFrame 1: \n {diferencia_2}')



##CONTADOR DE FILAS Y COLUMNAS DE DOS DATAFRAME
def contador_filas_columns(df_1,df_2):
    """
    Permite contar el numero de filas y de columnas del Data Frame 1 y 2.
	    Parametros:
		    df_1: DataFrame 2
            df_2: DataFrame 2
            
        Retorno:
		    Devuelve la cantidad de registros en Data Frame 1 y 2.
	"""	
    
    #Observamos la cantidad de filas y de columnas que hay en los dos datos de los Data Frame ya procesados
    df_1_shape = df_1.shape
    df_2_shape = df_2.shape

    print(f'La cantidad de registros en Data Frame 1:\n filas, columnas: {df_1_shape}')
    print(f'La cantidad de registros en Data Frame 2:\n filas, columnas: {df_2_shape}')



def generar_graficos_comparacion(diccionario_modelos, X_test, y_test, label_encoder, n_cols = 2):
    """
    Función especifica para generar los graficos de comparación de modelos.
    Input:
        diccionario_modelos: dict, diccionario con los nombres de los modelos y sus objetos correspondientes {nombre:objeto}
        X_test: dataframe con los datos a predecir
        y_test: dataframe con la variable objetivo
        label_encoder: objeto de codificacion de etiquetas
        n_cols: int, numero de columnas para generar los graficos
    Output:
        None: imprime los graficos, pero la funcion no retorna nada
    """
    ############################################
    # PROCESO PARA GENERAR DATAFRAME DE METRICAS
    ############################################
    dict_clases = dict(zip(range(0,len(list(label_encoder.classes_))),list(label_encoder.classes_)))
    dict_metricas = {
        'modelo': [],
        'cod_categoria' : [],
        'nom_categoria' : [],
        'metrica': [],
        'valor': []
    }
    for nombre,modelo in diccionario_modelos.items():
        y_pred = modelo.predict(X_test)
        for cod_cat in range(7):
            # se aplica un enfoque one vs rest
            y_pred_cat = [1 if x == cod_cat else 0 for x in y_pred] 
            y_test_cat = [1 if x == cod_cat else 0 for x in y_test] 
            
            for metrica in [accuracy_score, f1_score, precision_score, recall_score]:
                dict_metricas['modelo'].append(nombre)
                dict_metricas['cod_categoria'].append(cod_cat)
                dict_metricas['nom_categoria'].append(dict_clases[cod_cat])
                dict_metricas['metrica'].append(metrica.__name__)
                dict_metricas['valor'].append(metrica(y_test_cat, y_pred_cat))


    df_metricas = pd.DataFrame(dict_metricas)

    ######################
    # PROCESO DE GRAFICADO
    ######################

    rows = int(np.ceil(len(dict_clases)/n_cols))
    height = 2 * rows
    width = n_cols * 5

    fig = plt.figure(figsize=(width, height))
    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for cod,nom in dict_clases.items():
        datos = df_metricas[df_metricas['cod_categoria']==cod]

        plt.subplot(rows, n_cols, cod + 1)
        for i,modelo in enumerate(diccionario_modelos.keys()):
            plt.scatter(data=datos[datos['metrica'] == 'accuracy_score'] , x='valor', y='modelo', color=colors[0])
            plt.scatter(data=datos[datos['metrica'] == 'f1_score']       , x='valor', y='modelo', color=colors[1])
            plt.scatter(data=datos[datos['metrica'] == 'precision_score'], x='valor', y='modelo', color=colors[2])
            plt.scatter(data=datos[datos['metrica'] == 'recall_score']   , x='valor', y='modelo', color=colors[3])
        
        #plt.xlim((0.8, 1.0))
        plt.ylim((-1, len(diccionario_modelos.keys()) ))
        plt.yticks( range(0,len(diccionario_modelos.keys())) , list(diccionario_modelos.keys()) )
        plt.title(nom)

    fig.legend(['accuracy_score', 'f1_score','precision_score','recall_score'],loc = 'lower right')     
    fig.tight_layout()


class Modelo():
    """
    Clase creada para ser utilizada ne la interface de gradio
    no permite pasarle parametros adicionales a la funcion, por lo que esta funcion make_prediction
    los tomara directamente del objeto
    """
    def __init__(self,modelo,count_vectorizer,label_encoder,probabilidad_corte, funcion_predict, cantidad_palabras_min):
        """
        funcion inicializadora del objeto
        """
        self.modelo = modelo
        self.count_vectorizer = count_vectorizer
        self.label_encoder = label_encoder
        self.funcion_predict = funcion_predict
        self.probabilidad_corte = probabilidad_corte
        self.cantidad_palabras_min = cantidad_palabras_min

    def make_prediction(self, glosa):
        """
        Funcion que clasifica una glosa de denuncia
        input:
            glosa: txt, texto de la denuncia
        output:
            return: diccionario con las clases de tipo de delito y su probabilidad
        """
        preds = self.funcion_predict(self.modelo, self.count_vectorizer, self.label_encoder, [glosa])
        texto = self.label_encoder.classes_[list(preds[2][0]).index(max(preds[2][0]))]
        diccionario = dict(zip(self.label_encoder.classes_,preds[2][0]))
        
        if len(glosa.split()) < self.cantidad_palabras_min:
            texto = f"Favor ingresar más antecedentes (mínimo {self.cantidad_palabras_min} palabras)"
            diccionario = {}

        return texto,diccionario



def generar_countplot(dataframe, campo_x, label_encoder=None, procesar=True):
    """
    Genera un grafico de barras con la cantidad de elementos por categoria
    Input:
        dataframe: Dataframe con la data que se desea graficar
        campo_x: str, nombre de la variable categorica
        label_encoder: objeto con la codificacion de las clases si las clases son numericas
        procesar: bool, bandera para definir si realizar el proceso o no
    Output:
        None: imprime el grafico, pero no retorna nada
    """
    if procesar:
        # creamos un nuevo dataframe con los labels de las clases
        df_temp_labels = dataframe.copy()
        if df_temp_labels[campo_x].dtype in (np.int32,np.int64,np.float64) or type(df_temp_labels.at[0,campo_x]) in (int,float) or (df_temp_labels.at[0,campo_x]).isnumeric():
            df_temp_labels[campo_x] = df_temp_labels[campo_x].apply(lambda x: label_encoder.classes_[int(x)])
        
        # Graficamos la cantidad de datos en cada categoría del vector objetivo, ordenados por cantidad.
        plt.figure(figsize=(15,5))
        sns.countplot(data=df_temp_labels, x=campo_x, order=df_temp_labels[campo_x].value_counts().index)
        # mostramos la cantidad de datos en cada categoría
        for p in plt.gca().patches:
            plt.gca().annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()))
        plt.title('Cantidad de categorías de denuncias')
        plt.show()
        plt.close()



if __name__ == "__main__":
    pass