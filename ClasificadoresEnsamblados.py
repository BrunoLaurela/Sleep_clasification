from gssc.infer import EEGInfer
import mne
import yasa
import pandas as pd
import torch
import numpy as np
#######################
#estos clasificadores deben tener como entrada una rchivo raw y un archivo de metadatos que indique almenos señales de eeg, emg, eog
# la salida debe ser siempre las anotaciones y sus asociado pesos, que representan el porcentaje con el que se clasifico cada etapa del sueño
# las clasificaciones deben estar indicadas como [0,1,2,3,4], para wake nrem1 nrem2 nrem3 rem respectivaqmente
# los pesos debe ser una rray de 5*NdeEtapas, del tipo Pandas de DataFrame, debe idicarse en cada columna los indices de 'W' 'N1', 'N2' , 'N3', 'R'
# los pesos ademas deben estar entre 0 y 100, y no entre 0 y 1


#######
#GSSC
#######
def ClassifGSSC(raw, metadata):

    """
    Este método ejecuta el GSSC de manera automática usando los datos raw y los metadatos necesarios.
    
    Es necesario que los canales estén configurados correctamente con su tipo. Si no lo están, se puede aplicar 
    la función set_channels_type al objeto raw de MNE.

    Args:
        raw (mne): Datos raw de la polisomnografía.
        metadata (dict): Diccionario con todos los metadatos necesarios.

    Returns:
        pandas.DataFrame, array: Retorna un DataFrame de Pandas con los pesos calculados para cada etapa del sueño
                                y un array con las anotaciones predichas.

    El parámetro `raw` debe ser un objeto raw de MNE que contenga los datos de la polisomnografía.
    El parámetro `metadata` debe ser un diccionario que contenga los siguientes campos:
        - "channels": Un subdiccionario que contiene los nombres de los canales EEG y EOG.
                      Por ejemplo: metadata["channels"]["eeg"], metadata["channels"]["eog"].
        - ********* agregar mas**********
    """
    
    
    Net = EEGInfer()

    #raw.set_channel_types({ metadata["channels"]["eog"][:]: 'eog', metadata["channels"]["eeg"][:]: 'eeg'})  # esto voy a intentar hacer en el wrapper
    alleeg = []
    alleeg.extend(metadata['channels']['eeg']['central'])
    alleeg.extend(metadata['channels']['eeg']['frontal'])
    alleeg.extend(metadata['channels']['eeg']['parietal'])
    alleeg.extend(metadata['channels']['eeg']['occipital'])

    anotaciones,tiempos, min_logits = Net.mne_infer(inst = raw, eeg=alleeg, eog=metadata["channels"]["eog"])

    """
    min_logits tensor de torch, posee los pesos de cada clase, donde aquellos pesos mas relevante son mas sercanos a 0 esta matriz es creada por gssc
    aqui Transofrmo los valores de min logist a valores  del 0 al 100, correspondientes a el peso de cada etapa para el clasificador. 
    pesos es un onjetito tipo  torch.tensor de 5 columans y numero de filas como etapas del sueño, el cual posee los pesos en porcentaje del 0 al 100 para cada etapa
    """
    # Invertir los valores
    min_logis_inv = (1/min_logits)*-1   # dado que los valores son negativos multiplico por -1

    # Calcular la suma total de los valores invertidos
    suma = torch.sum(min_logis_inv, axis = 1)

    # Calcular el porcentaje para cada número invertido
    pesos = (min_logis_inv / suma.unsqueeze(1)) * 100

    pdPesos = pd.DataFrame(np.array(pesos), columns = ['W', 'N1', 'N2', 'N3', 'R'])

    return pdPesos, anotaciones

#####
#YASA
#####


def ClassifYASA(raw, metadata):
    """
    Realiza la clasificación del sueño utilizando YASA (Yet Another Sleep Staging Algorithm).

    Args:
        raw (object): Objeto que representa la señal EEG sin procesar.
        metadata (dict): Diccionario que contiene información sobre los canales de la señal, 
                         incluyendo los nombres de los canales EEG, EOG y EMG. entre otas cosas...

    Returns:
        tuple: Una tupla que contiene dos elementos:
            - pesos (array): Probabilidades de clasificación de cada epoch en cada estado de sueño.
            - anotaciones (array): Clasificaciones de sueño predichas para cada epoch.

    El parámetro `raw` debe contener la señal EEG sin procesar.
    El parámetro `metadata` debe ser un diccionario que contenga los siguientes campos:  
        - "channels": Un subdiccionario que contiene los nombres de los canales EEG, EOG y EMG.
                      Por ejemplo: metadata["channels"]["eeg"], metadata["channels"]["eog"], metadata["channels"]["emg"].
        - *********** seguir agregando *********** 
    
    """
    ## con esta linea de codigo me aseguro que si esta vacio el central uso el canal
    if metadata["channels"]["eeg"]["central"]:
        meta = metadata["channels"]["eeg"]["central"]
    elif metadata["channels"]["eeg"]["frontal"]:
        meta = metadata["channels"]["eeg"]["frontal"]
    else:
        meta = metadata["channels"]["eeg"]["parietal"]


   

    sls = yasa.SleepStaging(raw, eeg_name= meta[0], eog_name=metadata["channels"]["eog"][0], emg_name=metadata["channels"]["emg"][0])
    anotaciones = yasa.hypno_str_to_int(sls.predict())
    pesos = sls.predict_proba()
   
    
    return pesos*100, anotaciones # multiplico por 100 para que se mueva 2 decimales



    