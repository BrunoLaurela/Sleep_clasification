from ClasificadoresEnsamblados  import ClassifGSSC, ClassifYASA
import mne

#### metodo wrapper #####

def Entrada_clasificador(classif):  # sleep_stage_classification
    """
    Funcion envolvente de metodos de clasificacion del sueño
    La idea esque use esta funcion para chequear que la metadata este bien seteada 
    Puedo pasar distintos clasificadores del sueño, la unica condicion  esque estos clasificadores deben poseer como entrada :
    Un archivo raw de mne y metadata asociada, la salida  deben ser los pesos( es aquel valor que indica la "confianza" o "seguridad" con la cual se clasifica una etapa correspondiente)
    y las anotaciones asociadas.

    Args:
        classif (_type_): _description_
    """
    
    
    def wrapper(raw, metadata) : 
        #####
        # chequeo si  la metadata esta bien, modifico raw en caso de ser necesario
        #####

        #pesos, anotaciones = x
        
        return classif(raw, metadata)

    return wrapper


############### esta funcion me sirve solo para evaluar las etapas por separado ###################
def sleep_stage_classification(raw, metadata ):
    """_summary_

    Args:
        raw (_type_): _description_
        metadata (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    CLasificador1 = Entrada_clasificador(ClassifGSSC)
    CLasificador2 = Entrada_clasificador(ClassifYASA)
    
    pesos, anotaciones = CLasificador1(raw, metadata)
    pesos_2, anotaciones_2 = CLasificador2(raw, metadata) 



    return pesos, anotaciones,  pesos_2, anotaciones_2
##############################################
def AASM_Rules():
    """
    _summary_
    
    Aplico las Reglas del estandar y criterios de evaluacion para generar una clasificacion

    """
    pass

    return None
def Final_Vote(results):
    """_summary_

    Args:
        results (dic): diccionario donde la key es el nombre del clasificador, el primer elemento del diccionario
        son los pesos para cada clase y el 2do elemento son las anotaciones de la etapa del sueño
    """

    
   

    for name in results.keys():
        if name == 'GSSC':
            pesos_gssc  = results[name][0]
            anotaciones_gssc = results[name][1]
        if name =='YASA' :
            pesos_yasa  = results[name][0]
            anotaciones_yasa = results[name][1]
    
    No_concidencias = [1 if epoch_gssc != epoch_yasa else 0 for epoch_gssc, epoch_yasa in zip(anotaciones_gssc, anotaciones_yasa)]
    Candidatos_a_revision = No_concidencias
    
    if sum(No_concidencias) > sum(anotaciones_gssc)*0.3 :  # en estas lineas de codigo se implementa la estrategia de usar los pesos de gssc
        # para epocas candidatas a revision, solo se hara uso de esta estrategia cuando la implemntacin de las no coincidencias
        # de un numero mayor al 0.3 del raw completo dele estudio
        
        # Definir los criterios de incertidumbre
        criteria = {
            'W': 98,
            'N1': 0,
            'N2': 96,
            'N3': 97,
            'R': 96
        }

        # Crear una lista para almacenar los resultados de incertidumbre
        incertidumbre = []

        # Iterar sobre cada fila del DataFrame
        for index, row in pesos_gssc.iterrows():
            # Encontrar la columna con el valor máximo
            max_col = row.idxmax(axis=0)
            # Comprobar si el valor máximo cumple con el criterio de incertidumbre
            if row[max_col] < criteria[max_col]:
                incertidumbre.append(100)
            else:
                incertidumbre.append(0)

        Candidatos_a_revision = incertidumbre

        ################# ME FALTA GENERAQR EL CANAL STIM CON ELA RCHIVO Y PROBAR ESTA FUNCION POR SEPARADO ####################  
    # devuelve el archivo raw con un canal de stim donde se indica cuales eventos 
    #fueron detectados con poca presicion en ambois modelos de acuerdoa  algun criterio
    
  # genero un canal stim y agrego los eventos
    """
        pre_stim =np.zeros_like(np.array(np.arange(raw.n_times))) #[:,:][0][0]
        pre_stim[onsets_] = 1
        stim_chan = pre_stim.reshape(1,-1)

        mask_info = mne.create_info(ch_names=["STIMcomplexK"],
                                    sfreq=raw.info["sfreq"],
                                    ch_types=["stim"]
                                )
        raw_mask = mne.io.RawArray(data=stim_chan,
                                info=mask_info,
                                first_samp=raw.first_samp
                                )
        raw.add_channels([raw_mask], force_update_info=True)
    """
    d =3
    return d 
def classify_file(file_data, metadata, classifiers):
    results = {}
    for name, classifier in classifiers.items():
        result = classifier(file_data, metadata)
        results[name] = result
    return results
def sleep_stage_classification_file(file_data, metadata, classifiers, detectores):
    """_summary_

    Args:
        file_data (mne.raw): archivo raw de mne que contiene la señal de polisomnografia del estudio
        metadata (dict):diccionario con variables necesarias para hacer la clasiicacion
        tiene que tener el siguiente formato :
        dict = {
        channels :{ 'eog':[], 'eeg':{'frontal':[], 'central':[], 'parietal':[]},'emg': []} 
        }
        classifiers (dict): dicionario con los nombres y laos clasificadores que voy a usar  para clasificar las etapas del sueño
        debe tener el siguiente formato
        dict _ {
        classif : []
        }
        detectores (dict): dicionario con los nombres y las detectores que voy a usar para detectar eventos particulas del sueño
        debe tener el siguiente formato
        dict _ {
        detectors : []
        }

    Returns:
        array: retorna un array con los valores de la clasificacion final
    """
    
    ### 1ra parte del pipeline
    results = classify_file(file_data, metadata, classifiers)

    raw = Final_Vote(file_data,results)  # devuelve el archivo raw con un canal de stim donde se indica cuales eventos 
                                            #fueron detectados con poca presicion en ambois modelos de acuerdoa  algun criterio
    
    ### 2da parte del pipeline
    prediccion = AASM_Rules(raw,detectores)


    return prediccion