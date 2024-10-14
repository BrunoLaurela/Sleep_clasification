from ClasificadoresEnsamblados  import ClassifGSSC, ClassifYASA
from DetectoresEnsamblados import SpindleDetect,RemDetect,DetectorSW,Periodograma_Welch_por_segmento
import mne
import numpy as np
import pandas as pd
import yasa

#### metodo wrapper  no estoy usando esto .... BORRAR........#####

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

##############################################

def evaluacion_stim_channel(raw, channel_name, epoch_duration=30):
    """
    Evalúa los eventos de un canal específico en un conjunto de datos EEG.

    Esta función toma un objeto de datos EEG de MNE, selecciona un canal específico, 
    y evalúa los eventos en dicho canal para determinar en qué épocas (segmentos de tiempo)
    ocurrieron. Las épocas se determinan dividiendo el tiempo total de la señal por 
    una duración específica de época.

    Args:
        raw (mne.io.Raw): Objeto Raw de MNE que contiene los datos EEG.
        channel_name (str): Nombre del canal que se utilizará para la evaluación de eventos.
        epoch_duration (int, opcional): Duración de cada época en segundos. Por defecto es 30 segundos.

    Returns:
        pd.Series: Serie de pandas con la reevaluación de cada época. Cada índice representa una época,
                   y los valores representan el tiempo en segundos en el que ocurrió un evento dentro de esa época.
    """
    # Crear una copia del objeto raw y seleccionar el canal de interés
    raw_copy = raw.copy()
    raw_copy.pick_channels([channel_name])
    
    # Crear una serie de ceros con longitud igual al número de épocas
    Reevaluacion = np.zeros_like(np.arange(int(raw.n_times/raw.info['sfreq'])/epoch_duration))
    index = list(range(1, len(Reevaluacion)+1))
    Reevaluacion_pd = pd.Series(Reevaluacion, index=index)
    
    # Extraer eventos del canal de eventos
    events = mne.find_events(raw_copy)
    
    # Obtener las épocas en las que ocurrieron los eventos
    arrays = (events[:, 0] / raw.info['sfreq'] / epoch_duration).astype(int)
    muestras = (events[:, 0] / raw.info['sfreq']).astype(int)
    
    # Rellenar Reevaluacion_pd con las muestras ocurridas en cada época
    for id, muestra in zip(arrays, muestras):
        Reevaluacion_pd.loc[id] = muestra
    
    return Reevaluacion_pd
def ASSM_RulesDirectConReevaluacion(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]
    RMS_W = rms_list[EpocasW]
    RMS_R = rms_list[EpocasR]
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 5)

    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        
        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        
    
        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) == 0 or len(eventos_en_rango_sw) == 0 or len(eventos_en_rango_EpocaAnterior_spindle) == 0 or len(eventos_en_rango_EpocaAnterior_sw) == 0:
                #if  result['GSSC'][1][epoca-1] == 0:
                #    print('ES N1')
                #    Nueva_anotacion.append(1)
                if (Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha) and RMSenEpocaActual>N2q90rmsEEG:
                    print('ES N1')
                    Nueva_anotacion.append(1)
                #elif Porcentaje_de_densidad_Delta_oc < Q25_oc_MaxN2_Delta or Porcentaje_de_densidad_Delta_fr < Q25_fr_MaxN2_Delta or Porcentaje_de_densidad_Delta_cr < Q25_cr_MaxN2_Delta or Porcentaje_de_densidad_Delta_pr < Q25_pr_MaxN2_Delta:
                #    print('ES N1')
                #    Nueva_anotacion.append(1)
                else:
                    print("No se cumplieron las condiciones específicas para N1, queda como N2")
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)
        # este elife se usa la rpediccion de yasa pero para que no se confunda con REM que suele confundirse cuando clasifica N1 y para asegurarse N1 correctos se una Fr en alpha y RSM
        elif result['YASA'][1][epoca] == 1 and( Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha) and RMSenEpocaActual>N2q90rmsEEG :
            Nueva_anotacion.append(1)
        
        elif etapa == 0:
            # Aca me fijo si la etapa evaluada como W realmente corresponde a W
            if epoca + 1 <  result['GSSC'][1].shape[0]:
                #if  result['GSSC'][1][epoca + 1] == 2:
                #    Nueva_anotacion.append(1)
                #    print('ES N1')
                if (Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW)  and RMSenEpocaActual < Wq5rmsEEG:
                    Nueva_anotacion.append(1)
                    print('ES N1')
                else:
                    print("No se cumplieron las condiciones específicas para W, queda como W")
                    Nueva_anotacion.append(0)
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
                    


    return Nueva_anotacion
def ASSM_RulesPrueba12(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        
        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) == 0 or len(eventos_en_rango_sw) == 0 or len(eventos_en_rango_EpocaAnterior_spindle) == 0 or len(eventos_en_rango_EpocaAnterior_sw) == 0:
                #if  result['GSSC'][1][epoca-1] == 0:
                #    print('ES N1')
                #    Nueva_anotacion.append(1)
                if Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha:
                    print('ES N1')
                    Nueva_anotacion.append(1)
                #elif Porcentaje_de_densidad_Delta_oc < Q25_oc_MaxN2_Delta or Porcentaje_de_densidad_Delta_fr < Q25_fr_MaxN2_Delta or Porcentaje_de_densidad_Delta_cr < Q25_cr_MaxN2_Delta or Porcentaje_de_densidad_Delta_pr < Q25_pr_MaxN2_Delta:
                #    print('ES N1')
                #    Nueva_anotacion.append(1)
                else:
                    print("No se cumplieron las condiciones específicas para N1, queda como N2")
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)

        elif etapa == 0:
            # Aca me fijo si la etapa evaluada como W realmente corresponde a W
            if epoca + 1 <  result['GSSC'][1].shape[0]:
                if  result['GSSC'][1][epoca + 1] == 2:
                    Nueva_anotacion.append(1)
                    print('ES N1')
                elif Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW:
                    Nueva_anotacion.append(1)
                    print('ES N1')
                else:
                    print("No se cumplieron las condiciones específicas para W, queda como W")
                    Nueva_anotacion.append(0)
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
                    



    return Nueva_anotacion
def AASM_RulesPrueba14(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]
    RMS_W = rms_list[EpocasW]
    RMS_R = rms_list[EpocasR]
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 5)

    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            elif (Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha) : #and RMSenEpocaActual>N2q90rmsEEG:
                print('ES N1')
                Nueva_anotacion.append(1)
            else:
                Nueva_anotacion.append(2)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
          
            if  RMSenEpocaActual < Wq5rmsEEG:
                Nueva_anotacion.append(1)
                print('ES N1')
            elif (Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW):  
                Nueva_anotacion.append(1)
                print('ES N1')
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
           

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    # paso por una revision mas del codigo apra encontrar aquellos valores que capaz estaban como N2 pero debian ser N1
    
                


    return Nueva_anotacion
def AASM_RulesPrueba15(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]
    RMS_W = rms_list[EpocasW]
    RMS_R = rms_list[EpocasR]
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 5)

    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 90))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 90))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 90))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            else:
                Nueva_anotacion.append(1)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
          
            if  RMSenEpocaActual < Wq5rmsEEG and (Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW):
                Nueva_anotacion.append(1)
                print('ES N1')
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
           

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    # paso por una revision mas del codigo apra encontrar aquellos valores que capaz estaban como N2 pero debian ser N1
    
                


    return Nueva_anotacion
def AASM_RulesPrueba19(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]

    Chequeo_siEntro  = False
    """
    if all(x == 0 for x in prediccion[0:4]): # verifico que almenos alla 3 indices iguales a 0 para hacer esto
        index_of_first_non_zero = next((i for i, x in enumerate(prediccion) if x != 0), len(prediccion))
        Chequeo_siEntro = True
        # Calcula cuántos ceros iniciales hay
        initial_zeros = prediccion[:(index_of_first_non_zero-1)]

        # Toma el 80% de los ceros iniciales
        num_zeros_to_take = int(0.8 * len(initial_zeros))    
        RMS_W = rms_list[:num_zeros_to_take]
    else :
    """
    RMS_W = rms_list[EpocasW]

    RMS_R = rms_list[EpocasR]
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 15)
    PesosN2Q10 = np.nanpercentile(result['GSSC'][0]['N2'][EpocasN2], 5) # pongo 10 porque es la menor cantidad de epcoas con las que suele confundirse con N1

    ##############################################################
        # EVALUACION DE PESOS #
    if all(x == 0 for x in prediccion[0:4]): # verifico que almenos alla 3 indices iguales a 0 para hacer esto
        index_of_first_non_zero = next((i for i, x in enumerate(prediccion) if x != 0), len(prediccion))
        Chequeo_siEntro = True
        # Calcula cuántos ceros iniciales hay
        initial_zeros = prediccion[:(index_of_first_non_zero-1)]

        # Toma el 80% de los ceros iniciales
        num_zeros_to_take = int(0.8 * len(initial_zeros))   
        mediana_pesosW = np.median(result['GSSC'][0]['W'][:num_zeros_to_take])
        mediana_pesosN2 =np.median(result['GSSC'][0]['N2'][EpocasN2])
    else :
        mediana_pesosW = 0  # poner mediana de pesos = 0 quiere decir que desitimo esta variable si es que no tengo suficientes epocas apra calcular la mediana en W
        mediana_pesosN2 = 0
    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################### PESO EPOCA ACTUAL ##################################
        peso_actual = result['GSSC'][0]['W'][epoca]
        peso_actualN2 = result['GSSC'][0]['N2'][epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            elif Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha:
                print('ES N1')
                Nueva_anotacion.append(1)
            elif peso_actualN2 < PesosN2Q10 :
                Nueva_anotacion.append(1)
            elif (epoca - 1 >= 0) and (epoca + 1 < result['GSSC'][1].shape[0]):
                # Verificar si la época anterior y posterior son ambas iguales a 1
                if result['GSSC'][1][epoca - 1] == 1 and result['GSSC'][1][epoca + 1] == 1:
                    # Si ambas épocas son 1, hacer algo
                    Nueva_anotacion.append(1)
                else: 
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
            if (RMSenEpocaActual < Wq5rmsEEG )  and ( peso_actual < mediana_pesosW ) :
                Nueva_anotacion.append(1)
                print('ES N1')
        
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
        

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    # paso por una revision mas del codigo apra encontrar aquellos valores que capaz estaban como N2 pero debian ser N1
    ##################### Evaluacion de los N1 en N2 ##########################
    #Prediccion_correccion = [0 if i != 0 and i != len(Nueva_anotacion) - 1 and Nueva_anotacion[i + 1] == 0 and Nueva_anotacion[i - 1] == 0 else v for i, v in enumerate(Nueva_anotacion)]

    return Nueva_anotacion
def AASM_RulesPrueba17(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]

    Chequeo_siEntro  = False
    """
    if all(x == 0 for x in prediccion[0:4]): # verifico que almenos alla 3 indices iguales a 0 para hacer esto
        index_of_first_non_zero = next((i for i, x in enumerate(prediccion) if x != 0), len(prediccion))
        Chequeo_siEntro = True
        # Calcula cuántos ceros iniciales hay
        initial_zeros = prediccion[:(index_of_first_non_zero-1)]

        # Toma el 80% de los ceros iniciales
        num_zeros_to_take = int(0.8 * len(initial_zeros))    
        RMS_W = rms_list[num_zeros_to_take]
    else :
    """
    RMS_W = rms_list[EpocasW]

    RMS_R = rms_list[EpocasR]
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 10)

    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            elif (epoca - 1 >= 0) and (epoca + 1 < result['GSSC'][1].shape[0]):
                # Verificar si la época anterior y posterior son ambas iguales a 1
                if result['GSSC'][1][epoca - 1] == 1 and result['GSSC'][1][epoca + 1] == 1:
                    # Si ambas épocas son 1, hacer algo
                    Nueva_anotacion.append(1)
                else: 
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
          
            if  RMSenEpocaActual < Wq5rmsEEG :
                Nueva_anotacion.append(1)
                print('ES N1')
            elif  (Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW):
                Nueva_anotacion.append(1)
                print('ES N1')
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
           

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    # paso por una revision mas del codigo apra encontrar aquellos valores que capaz estaban como N2 pero debian ser N1
    
                


    return Nueva_anotacion,Chequeo_siEntro
def AASM_RulesPrueba20(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]
    RMS_W = rms_list[EpocasW]
    RMS_R = rms_list[EpocasR]

    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 15)
    PesosN2Q10 = np.nanpercentile(result['GSSC'][0]['N2'][EpocasN2], 15) # pongo 10 porque es la menor cantidad de epcoas con las que suele confundirse con N1

    ##############################################################
        # EVALUACION DE PESOS #
    if all(x == 0 for x in prediccion[0:4]): # verifico que almenos alla 3 indices iguales a 0 para hacer esto
        index_of_first_non_zero = next((i for i, x in enumerate(prediccion) if x != 0), len(prediccion))
        Chequeo_siEntro = True
        # Calcula cuántos ceros iniciales hay
        initial_zeros = prediccion[:(index_of_first_non_zero-1)]

        # Toma el 80% de los ceros iniciales
        num_zeros_to_take = int(0.8 * len(initial_zeros))   
        PesosWQ15= np.nanpercentile(result['GSSC'][0]['W'][:num_zeros_to_take], 15)
        mediana_pesosN2 = np.nanpercentile(result['GSSC'][0]['N2'][EpocasN2],10)
    else :
        PesosWQ15 = 0  # poner mediana de pesos = 0 quiere decir que desitimo esta variable si es que no tengo suficientes epocas apra calcular la mediana en W
        mediana_pesosN2 = 0
    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 85))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 85))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 85))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################### PESO EPOCA ACTUAL ##################################
        peso_actual = result['GSSC'][0]['W'][epoca]
        peso_actualN2 = result['GSSC'][0]['N2'][epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            elif Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha:
                print('ES N1')
                Nueva_anotacion.append(1)
            elif peso_actualN2 < PesosN2Q10 :
                Nueva_anotacion.append(1)
            elif np.sum(duracion__en_rango_sw) > 0.2*30:
                Nueva_anotacion.append(3)

            elif (epoca - 1 >= 0) and (epoca + 1 < result['GSSC'][1].shape[0]):
                # Verificar si la época anterior y posterior son ambas iguales a 1
                if result['GSSC'][1][epoca - 1] == 1 and result['GSSC'][1][epoca + 1] == 1:
                    # Si ambas épocas son 1, hacer algo
                    Nueva_anotacion.append(1)
                else: 
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
            if (RMSenEpocaActual < Wq5rmsEEG )  and ( peso_actual < PesosWQ15 ) :
                Nueva_anotacion.append(1)
                print('ES N1')
        
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
        

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    
    return Nueva_anotacion
def AASM_RulesPrueba16(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]
    RMS_W = rms_list[EpocasW]
    RMS_R = rms_list[EpocasR]
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 5)

    ##############################################################

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################
        
        

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            elif (epoca - 1 >= 0) and (epoca + 1 < result['GSSC'][1].shape[0]):
                # Verificar si la época anterior y posterior son ambas iguales a 1
                if result['GSSC'][1][epoca - 1] == 1 and result['GSSC'][1][epoca + 1] == 1:
                    # Si ambas épocas son 1, hacer algo
                    Nueva_anotacion.append(1)
                else: 
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
          
            if  RMSenEpocaActual < Wq5rmsEEG and (Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW):
                Nueva_anotacion.append(1)
                print('ES N1')
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
           

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    # paso por una revision mas del codigo apra encontrar aquellos valores que capaz estaban como N2 pero debian ser N1
    
                


    return Nueva_anotacion  
def AASM_RulesDirecto(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    Reevaluacion_pd = evaluacion_stim_channel(raw, channel_name ='STIM_SAMPLE_TO_EVALUATE')



    
    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar        
    reevaluacion = Reevaluacion_pd
    predicciones = result['GSSC'][1]
    if len(reevaluacion) > len(result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, deteccion in enumerate(Reevaluacion_pd): # epoca : indica la epoca en  que oruccurio al deteccion, deteccion : indica el momento en segundos en que ocurrio la deteccion
        # inicializo esta lista de puntos en 0, apa luego ir sumando en el indice correspondiente si hay mas chances de alguna epoca en especifico
        
        ###################### wake
        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else :  Porcentaje_de_densidad_alpha_oc = 0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else : Porcentaje_de_densidad_alpha_fr = 0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else : Porcentaje_de_densidad_alpha_cr  = 0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else : Porcentaje_de_densidad_alpha_pr  = 0
        ################################
        if deteccion!= 0 :
            
            #print('ingreso, en la epoca :', epoca)

            start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            # obtengo los eventos   de interes en la epoca evaluada
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
            eventos_en_rango_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            eventos_en_rango_REM = DeteccionRem[(DeteccionRem >= start_time) & (DeteccionRem < end_time) ]
            
            start_time = deteccion  - 30 # epoca anterior
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            condicion2 =(
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2] 
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start']  >= start_time) & 
                (eventosSW['Start']  < end_time)
            ]  
            if  np.sum(duracion__en_rango_sw) > 0.2*30:  
                print('ES N3') 
                Nueva_anotacion.append(3)                
            elif len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw)!= 0 :
                print('Es N2')
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_REM)!=0 :
                print('Es REM')
                Nueva_anotacion.append(4)
            elif Porcentaje_de_densidad_alpha_cr > 0.25 or  Porcentaje_de_densidad_alpha_fr> 0.25 or  Porcentaje_de_densidad_alpha_pr > 0.25  or Porcentaje_de_densidad_alpha_oc <0.25:
                print('Es wake')
                Nueva_anotacion.append(0)
            elif result['GSSC'][1][epoca-1] == 4 :   # si al epoca anterior fue rem
                print('Es REM')
                Nueva_anotacion.append(4)
            elif result['GSSC'][1][epoca-1] == 0 :   # si la epoca anterior fue Wake
                print('Es N1')
                Nueva_anotacion.append(1)
            else :
                print('queda como GSSC')
                Nueva_anotacion.append(result['GSSC'][1][epoca])
                
        else : 
             Nueva_anotacion.append(result['GSSC'][1][epoca])
               
    
    return Nueva_anotacion
def AASM_RulesDirecto_todos_conGSSC(raw, metadata, result):
    
    Nueva_anotacion = []
    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = np.ones(int(raw.n_times / raw.info['sfreq'] / 30), dtype=int)
   
    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar        
   
    predicciones = result['GSSC'][1]
    minlen = min(len(predicciones), len(Reevaluacion_pd))
    Reevaluacion_pd = Reevaluacion_pd[:minlen]
    predicciones = predicciones[:minlen]
    reevaluacion = Reevaluacion_pd
    predicciones = result['GSSC'][1]
    
    if len(reevaluacion) > len(result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, deteccion in enumerate(Reevaluacion_pd): # epoca : indica la epoca en  que oruccurio al deteccion, deteccion : indica el momento en segundos en que ocurrio la deteccion
        # inicializo esta lista de puntos en 0, apa luego ir sumando en el indice correspondiente si hay mas chances de alguna epoca en especifico
        ###################### wake
        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else :  Porcentaje_de_densidad_alpha_oc = 0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else : Porcentaje_de_densidad_alpha_fr = 0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else : Porcentaje_de_densidad_alpha_cr  = 0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
        else : Porcentaje_de_densidad_alpha_pr  = 0
        ################################
        if True :
            
            #print('ingreso, en la epoca :', epoca)

            start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            # obtengo los eventos   de interes en la epoca evaluada
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
            eventos_en_rango_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            eventos_en_rango_REM = DeteccionRem[(DeteccionRem >= start_time) & (DeteccionRem < end_time) ]
            
            start_time = deteccion  - 30 # epoca anterior
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            condicion2 =(
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2] 
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start']  >= start_time) & 
                (eventosSW['Start']  < end_time)
            ]  
            if  np.sum(duracion__en_rango_sw) > 0.2*30:  
                print('ES N3') 
                Nueva_anotacion.append(3)                
            elif len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw)!= 0 :
                print('Es N2')
                Nueva_anotacion.append(2)   
            elif len(eventos_en_rango_REM)!=0 :
                print('Es REM')
                Nueva_anotacion.append(4)
            elif Porcentaje_de_densidad_alpha_cr > 0.25 or  Porcentaje_de_densidad_alpha_fr> 0.25 or  Porcentaje_de_densidad_alpha_pr > 0.25  or Porcentaje_de_densidad_alpha_oc <0.25:
                print('Es wake')
                Nueva_anotacion.append(0)
            elif result['GSSC'][1][epoca-1] == 4 :
                print('Es REM')
                Nueva_anotacion.append(4)
            elif result['GSSC'][1][epoca-1] == 0 :
                print('Es N1')
                Nueva_anotacion.append(1)
            else :
                print('queda como GSSC')
                Nueva_anotacion.append(result['GSSC'][1][epoca])
                
        else : 
            Nueva_anotacion.append(result['GSSC'][1][epoca])
               
    
    return Nueva_anotacion   
def AASM_Rules(raw,metadata,result):

    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    """
    _summary_
    
    Aplico las Reglas del estandar y criterios de evaluacion para generar una clasificacion

    """
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    Reevaluacion_pd = evaluacion_stim_channel(raw, channel_name ='STIM_SAMPLE_TO_EVALUATE')

    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar        
    reevaluacion = Reevaluacion_pd
    predicciones = result['GSSC'][1]
    if len(reevaluacion) > len(result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, deteccion in enumerate(Reevaluacion_pd): # epoca : indica la epoca en  que oruccurio al deteccion, deteccion : indica el momento en segundos en que ocurrio la deteccion
        # inicializo esta lista de puntos en 0, apa luego ir sumando en el indice correspondiente si hay mas chances de alguna epoca en especifico
        Puntos = [0] * 5 # idx : 0 es W idx 1,2,3 es n1, n2, n3 respectivamente, idx 4 es REM
        
        if deteccion!= 0 :
            
            #print('ingreso, en la epoca :', epoca)

            start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            # obtengo los eventos   de interes en la epoca evaluada
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]

            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
            eventos_en_rango_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            eventos_en_rango_REM = DeteccionRem[(DeteccionRem >= start_time) & (DeteccionRem < end_time) ]
            
            start_time = deteccion  - 30 # epoca anterior
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            condicion2 =(
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2] 
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start']  >= start_time) & 
                (eventosSW['Start']  < end_time)
            ]  

        
            # STAGE W
            band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
            """
            2. Score epochs as stage W when more than 50% of the epoch has alpha rhythm over the
            #occipital region.
            """
            if metadata['channels']['eeg']['occipital']:
                for canal_O in  metadata['channels']['eeg']['occipital']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            # PRUEBA 2 - W
    
            if metadata['channels']['eeg']['frontal']:
                for canal_O in  metadata['channels']['eeg']['frontal']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            if metadata['channels']['eeg']['central']:
                for canal_O in  metadata['channels']['eeg']['central']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1

            if Puntos[0] == 0 :   # si no esta en ningun canal el ritmo alpha superior se suma puntos apra ser N1  -- Descarte N1 
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            """
            3. Score epochs without visually discernible alpha rhythm as stage W if ANY of the following are
            present:
            a. Eye blinks at a frequency of 0.5-2 Hz
            b. Reading eye movements
            c. Irregular, conjugate rapid eye movements associated wit normal or high chin muscle tone Slow eye movements (SEM): Conjugate, reasonably regular, sinusoidal eye movements with an initial deflection
            usually lasting >500 msec.   
            """
            """
            if len(eventos_en_rango_REM)!=0  & duracion_delEvento > estiamacion  & pendiente_de_las_ondas > estiamcion :  # tengo que considerar que en W se detectan movimientos de ojos conjugados, pero no son d emanera rapida
                print('Suma puntos para ser W') 
            if  # detectar amplitud alta del EMG 
            """
            #STAGE N1
            """

            In subjects who generate alpha rhythm, score stage N1 if the alpha rhythm is attenuated and
            replaced by low-amplitude, mixed-frequency activity for more than 50% of the
            epoch.N1,N2,N3
            """
            dendidad_espectral = []
            band_names = [ 'Theta', 'Alpha', 's'] # 'Delta', 'Beta' saco estas 2 pq supongo que en la frecuencia mixta no participan
            varianza_x_canales = []
            
            for canal_eeg in metadata['channels']['eeg']:
                if  metadata['channels']['eeg'][canal_eeg]:
                    for canal in  metadata['channels']['eeg'][canal_eeg]:
                        for banda in band_names:
                            Porcentaje_de_densidad = periodograma.loc[banda, canal][epoca] 
                            dendidad_espectral.append(Porcentaje_de_densidad)
                    varianza = np.var(dendidad_espectral)
                    varianza_x_canales.append(varianza)
            umbral_varianza = 0.1  # tipo por tirar ...  no sabria que umbral poner
            if np.var(varianza_x_canales) < umbral_varianza :
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if result['GSSC'][1][epoca-1] == 0: # osea indica que es wake la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if result['GSSC'][1][epoca-1] == 1: # osea indica que es N1 la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if result['GSSC'][1][epoca-1] == 4 :#osea indica que es  rem la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            """
            3. In subjects who do not generate alpha rhythm, score stage N1 commencing with the earliest of
            ANY of the following phenomena:N1,N2,N3,N4
            a. EEG activity in range of 4-7 Hz with slowing of background frequencies by ≥1 Hz from those of stage W
            b. Vertex sharp waves
            c. Slow eye movements
            """
            # STAGE 2
            """
            2. Begin scoring stage N2 (in absence of criteria for N3) if EITHER OR BOTH of the following occur
            during the first half of that epoch or the last half of the previous epoch:N1,N2,N3,N4
            a. One or more K complexes unassociated with arousals
            b. One or more trains of sleep spindles
            """
            ##### revisar este if
            if len(eventos_en_rango_spindle) != 0 or (len(eventos_en_rango_sw)!= 0 and np.sum(duracion__en_rango_sw) < 0.2*30):  # se considera que no se N3 para evalaur N2
                print('Suma puntos para ser N2')  
                                                                                            # se pone 0.2*30 para descartar que sea N3 dado que muchos kc juntos correspondena  etapa n3`
                Puntos[2] += 100 # osea si ocurre un complejo K o un spindle  se la considera Huso del sueño, por lo que se suma 100 puntos
                ### podria ser N3
            else :
                Puntos[1] += 1  # por descarte elijo a N1
                print('suma puntos para ser N1')
                
                """
                Continue to score epochs with low-amplitude, mixed-frequency EEG activity without K
                complexes or sleep spindles as stage N2 if they are preceded by epochs containing EITHER of the
                following:
                a. K complexes unassociated with arousals
                b. Sleep spindles
                """ 
            if len(eventos_en_rango_EpocaAnterior_spindle)!= 0 or len(eventos_en_rango_EpocaAnterior_sw)!= 0 :
                print('suma puntos para ser N2')
                Puntos[2] += 1
            if result['GSSC'][1][epoca-1] == 1 :# si la epoca anterior era N1 suma putos para que esta sea N2  - Consultar
                print('suma puntos para ser N2')
                Puntos[2] += 1
            if result['GSSC'][1][epoca-1] ==2: # osea indica que es N2 la epoca anterior - Consultar
                print('suma puntos para ser N2')
                Puntos[2] += 1

            # STAGE 3
            """
            Score stage N3 when ≥20% of an epoch consists of slow wave activity, irrespective of
            age.N2,N3,N4
            """
            
            if  np.sum(duracion__en_rango_sw) > 0.2*30 :
                print('suma puntos para ser N3')
                Puntos[3] += 100  # en la proxima ponerle 100 y..
            
            else :
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            if result['GSSC'][1][epoca-1] ==3: # osea indica que es N3 la epoca anterior   -Consultar
                print('suma puntos para ser N3')
                Puntos[3] += 1
            if result['GSSC'][1][epoca-1] ==2: # osea indica que es N2 la epoca anterior - Consultar
                print('suma puntos para ser N3')
                Puntos[3] += 1
            
            
            """
            Note 4. In stage N3, the chin EMG is of variable amplitude, often lower than in stage N2 sleep and sometimes as low
            as in stage R sleep.
            """
            # STAGE R
        
            """
            2. Score stage R sleep in epochs with ALL of the following phenomena:N1,N2,N3
            a. Low-amplitude, mixed-frequency EEG
            b. Low chin EMG tone
            c. Rapid eye movements
            """
            # capaz mejor que se vaya a REM 
            if len(eventos_en_rango_REM)!=0 :
                print('suma puntos para ser rem')
                Puntos[4] += 1

            varianza_x_canales = []    ##################### revisar igualq  mismo con n1
            for canal_eeg in metadata['channels']['eeg']:
                if  metadata['channels']['eeg'][canal_eeg]:
                    for canal in  metadata['channels']['eeg'][canal_eeg]:
                        for banda in band_names:
                            Porcentaje_de_densidad = periodograma.loc[banda, canal][epoca] 
                            dendidad_espectral.append(Porcentaje_de_densidad)
                    varianza = np.var(dendidad_espectral)
                    umbral_varianza = 0.1  # tipo por tirar ...  no sabria que umbral poner
                    if   varianza < umbral_varianza :
                            print('suma puntos para ser r')
                            Puntos[4] += 1
            """
            3. Continue to score stage R sleep, even in the absence of rapid eye movements, for epochs
            following one or more epochs of stage R as defined in rule I.2 above, IF the EEG continues to
            show low-amplitude, mixed-frequency activity without K complexes or sleep spindles AND the
            chin EMG tone remains low for the majority of the epoch.N4
            """
            if result['GSSC'][1][epoca-1] == 4 and (len(eventos_en_rango_spindle) != 0 or (len(eventos_en_rango_sw)!= 0)): # osea indica que  no hay usos del sueño ni spindle y al epoca anterior era rem
                print('suma puntos para ser rem')
                Puntos[4] += 1
            if result['GSSC'][1][epoca-1] == 1 : # osea indico que hay fase 3
                print('suma puntos para ser rem')
                Puntos[4] += 1
            
            elif Puntos[4] == 0 :
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            # creo esta nueva etapa  dependiendo del puntaje
            if np.std(Puntos) == 0 :
                epoca_elejida = 1 # osea si todas las anotaciones tienen la misma probabilidad se elije a N1 como candidata
            else :
                epoca_elejida = np.argmax(Puntos)
            Nueva_anotacion.append(epoca_elejida)
        else :
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    
    return Nueva_anotacion
def AASM_rules_todos_conGSSC(raw, metadata,result):
    
    Nueva_anotacion = []
    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = np.ones(int(raw.n_times / raw.info['sfreq'] / 30), dtype=int)
    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar        
    predicciones = result['GSSC'][1]
    minlen = min(len(predicciones), len(Reevaluacion_pd))
    Reevaluacion_pd = Reevaluacion_pd[:minlen]
    predicciones = predicciones[:minlen]


    for epoca, deteccion in enumerate(Reevaluacion_pd): # epoca : indica la epoca en  que oruccurio al deteccion, deteccion : indica el momento en segundos en que ocurrio la deteccion
        # inicializo esta lista de puntos en 0, apa luego ir sumando en el indice correspondiente si hay mas chances de alguna epoca en especifico
        Puntos = [0] * 5 # idx : 0 es W idx 1,2,3 es n1, n2, n3 respectivamente, idx 4 es REM
        
        if True :
            
            #print('ingreso, en la epoca :', epoca)

            start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            # obtengo los eventos   de interes en la epoca evaluada
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]

            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
            eventos_en_rango_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            eventos_en_rango_REM = DeteccionRem[(DeteccionRem >= start_time) & (DeteccionRem < end_time) ]
            
            start_time = deteccion  - 30 # epoca anterior
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            condicion2 =(
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2] 
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start']  >= start_time) & 
                (eventosSW['Start']  < end_time)
            ]  

        
            # STAGE W
            band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
            """
            2. Score epochs as stage W when more than 50% of the epoch has alpha rhythm over the
            #occipital region.
            """
            if metadata['channels']['eeg']['occipital']:
                for canal_O in  metadata['channels']['eeg']['occipital']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            # PRUEBA 2 - W
        
            if metadata['channels']['eeg']['frontal']:
                for canal_O in  metadata['channels']['eeg']['frontal']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            if metadata['channels']['eeg']['central']:
                for canal_O in  metadata['channels']['eeg']['central']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            if Puntos[0] == 0 :   # si no esta en ningun canal el ritmo alpha superior se suma puntos apra ser N1  -- Descarte N1 
                Puntos[1] = +1
                print('suma puntos para ser N1')

            
            """
            3. Score epochs without visually discernible alpha rhythm as stage W if ANY of the following are
            present:
            a. Eye blinks at a frequency of 0.5-2 Hz
            b. Reading eye movements
            c. Irregular, conjugate rapid eye movements associated wit normal or high chin muscle tone Slow eye movements (SEM): Conjugate, reasonably regular, sinusoidal eye movements with an initial deflection
            usually lasting >500 msec.   
            """
            """
            if len(eventos_en_rango_REM)!=0  & duracion_delEvento > estiamacion  & pendiente_de_las_ondas > estiamcion :  # tengo que considerar que en W se detectan movimientos de ojos conjugados, pero no son d emanera rapida
                print('Suma puntos para ser W') 
            if  # detectar amplitud alta del EMG 
            """
            #STAGE N1
            """

            In subjects who generate alpha rhythm, score stage N1 if the alpha rhythm is attenuated and
            replaced by low-amplitude, mixed-frequency activity for more than 50% of the
            epoch.N1,N2,N3
            """
            dendidad_espectral = []
            band_names = [ 'Theta', 'Alpha', 'Sigma'] # 'Delta', 'Beta' saco estas 2 pq supongo que en la frecuencia mixta no participan
            varianza_x_canales = []
            
            for canal_eeg in metadata['channels']['eeg']:
                if  metadata['channels']['eeg'][canal_eeg]:
                    for canal in  metadata['channels']['eeg'][canal_eeg]:
                        for banda in band_names:
                            Porcentaje_de_densidad = periodograma.loc[banda, canal][epoca] 
                            dendidad_espectral.append(Porcentaje_de_densidad)
                    varianza = np.var(dendidad_espectral)
                    varianza_x_canales.append(varianza)
            umbral_varianza = 0.1  # tipo por tirar ...  no sabria que umbral poner
            if np.var(varianza_x_canales) < umbral_varianza :
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if result['GSSC'][1][epoca-1] == 0: # osea indica que es wake la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if result['GSSC'][1][epoca-1] == 1: # osea indica que es N1 la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if result['GSSC'][1][epoca-1] == 4 :#osea indica que es  rem la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            """
            3. In subjects who do not generate alpha rhythm, score stage N1 commencing with the earliest of
            ANY of the following phenomena:N1,N2,N3,N4
            a. EEG activity in range of 4-7 Hz with slowing of background frequencies by ≥1 Hz from those of stage W
            b. Vertex sharp waves
            c. Slow eye movements
            """
            # STAGE 2
            """
            2. Begin scoring stage N2 (in absence of criteria for N3) if EITHER OR BOTH of the following occur
            during the first half of that epoch or the last half of the previous epoch:N1,N2,N3,N4
            a. One or more K complexes unassociated with arousals
            b. One or more trains of sleep spindles
            """
            
            if len(eventos_en_rango_spindle) != 0 or (len(eventos_en_rango_sw)!= 0 and np.sum(duracion__en_rango_sw) < 0.2*30):  # se considera que no se N3 para evalaur N2
                print('Suma puntos para ser N2')  
                                                                                            # se pone 0.2*30 para descartar que sea N3 dado que muchos kc juntos correspondena  etapa n3`
                Puntos[2] += 100 # osea si ocurre un complejo K o un spindle  se la considera Huso del sueño, por lo que se suma 100 puntos
                
            else :
                Puntos[1] += 1  # por descarte elijo a N1
                print('suma puntos para ser N1')
                
                """
                Continue to score epochs with low-amplitude, mixed-frequency EEG activity without K
                complexes or sleep spindles as stage N2 if they are preceded by epochs containing EITHER of the
                following:
                a. K complexes unassociated with arousals
                b. Sleep spindles
                """ 
            if len(eventos_en_rango_EpocaAnterior_spindle)!= 0 or len(eventos_en_rango_EpocaAnterior_sw)!= 0 :
                print('suma puntos para ser N2')
                Puntos[2] += 1
            if result['GSSC'][1][epoca-1] == 1 :# si la epoca anterior era N1 suma putos para que esta sea N2  - Consultar
                print('suma puntos para ser N2')
                Puntos[2] += 1
            if result['GSSC'][1][epoca-1] ==2: # osea indica que es N2 la epoca anterior - Consultar
                print('suma puntos para ser N2')
                Puntos[2] += 1

            # STAGE 3
            """
            Score stage N3 when ≥20% of an epoch consists of slow wave activity, irrespective of
            age.N2,N3,N4
            """
            
            if  np.sum(duracion__en_rango_sw) > 0.2*30 :
                print('suma puntos para ser N3')
                Puntos[3] += 100  # en la proxima ponerle 100 y..
            
            else :
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            if result['GSSC'][1][epoca-1] ==3: # osea indica que es N3 la epoca anterior   -Consultar
                print('suma puntos para ser N3')
                Puntos[3] += 1
            if result['GSSC'][1][epoca-1] ==2: # osea indica que es N2 la epoca anterior - Consultar
                print('suma puntos para ser N3')
                Puntos[3] += 1
            
            
            """
            Note 4. In stage N3, the chin EMG is of variable amplitude, often lower than in stage N2 sleep and sometimes as low
            as in stage R sleep.
            """
            # STAGE R
        
            """
            2. Score stage R sleep in epochs with ALL of the following phenomena:N1,N2,N3
            a. Low-amplitude, mixed-frequency EEG
            b. Low chin EMG tone
            c. Rapid eye movements
            """
            if len(eventos_en_rango_REM)!=0 :
                print('suma puntos para ser rem')
                Puntos[4] += 1

            varianza_x_canales = []    ##################### revisar igualq  mismo con n1
            for canal_eeg in metadata['channels']['eeg']:
                if  metadata['channels']['eeg'][canal_eeg]:
                    for canal in  metadata['channels']['eeg'][canal_eeg]:
                        for banda in band_names:
                            Porcentaje_de_densidad = periodograma.loc[banda, canal][epoca] 
                            dendidad_espectral.append(Porcentaje_de_densidad)
                    varianza = np.var(dendidad_espectral)
                    umbral_varianza = 0.1  # tipo por tirar ...  no sabria que umbral poner
                    if   varianza < umbral_varianza :
                            print('suma puntos para ser r')
                            Puntos[4] += 1
            """
            3. Continue to score stage R sleep, even in the absence of rapid eye movements, for epochs
            following one or more epochs of stage R as defined in rule I.2 above, IF the EEG continues to
            show low-amplitude, mixed-frequency activity without K complexes or sleep spindles AND the
            chin EMG tone remains low for the majority of the epoch.N4
            """
            if result['GSSC'][1][epoca-1] == 4 and (len(eventos_en_rango_spindle) != 0 or (len(eventos_en_rango_sw)!= 0)): # osea indica que  no hay usos del sueño ni spindle y al epoca anterior era rem
                print('suma puntos para ser rem')
                Puntos[4] += 1
            if result['GSSC'][1][epoca-1] == 1 : # osea indico que hay fase 3
                print('suma puntos para ser rem')
                Puntos[4] += 1
            
            elif Puntos[4] == 0 :
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            # creo esta nueva etapa  dependiendo del puntaje
            if np.std(Puntos) == 0 :
                epoca_elejida = 1 # osea si todas las anotaciones tienen la misma probabilidad se elije a N1 como candidata
            else :
                epoca_elejida = np.argmax(Puntos)
            Nueva_anotacion.append(epoca_elejida)
        else :
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    
    return Nueva_anotacion
def AASM_rules_todos(raw, metadata):
    
    ###################################################

    # Aqui se prueba otra variante de ASSM_rules, usando todoas las epocas apra reevalaur

    ###################################################

    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = np.ones(int(raw.n_times / raw.info['sfreq'] / 30), dtype=int)


   
    reevaluacion = Reevaluacion_pd
   
    Nueva_anotacion.append(0)  # inicializo la primera etapa como WAKE


    for epoca, deteccion in enumerate(Reevaluacion_pd[1:]): # epoca : indica la epoca en  que oruccurio al deteccion, deteccion : indica el momento en segundos en que ocurrio la deteccion
        # inicializo esta lista de puntos en 0, apa luego ir sumando en el indice correspondiente si hay mas chances de alguna epoca en especifico
        Puntos = [0] * 5 # idx : 0 es W idx 1,2,3 es n1, n2, n3 respectivamente, idx 4 es REM
        
        if deteccion!= 0 :
            
            deteccion = epoca*30
            start_time = epoca*30  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            # obtengo los eventos   de interes en la epoca evaluada
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]

            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
            eventos_en_rango_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            eventos_en_rango_REM = DeteccionRem[(DeteccionRem >= start_time) & (DeteccionRem < end_time) ]
            
            start_time = deteccion  - 30 # epoca anterior
            first_half_period = deteccion + 15
            previous_half_period = deteccion-15
            end_time = start_time + 30  
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[(eventosSpindle >= start_time) & (eventosSpindle < first_half_period) |(eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)] 
            condicion2 =(
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2] 
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start']  >= start_time) & 
                (eventosSW['Start']  < end_time)
            ]  

        
            # STAGE W
            band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
            """
            2. Score epochs as stage W when more than 50% of the epoch has alpha rhythm over the
            #occipital region.
            """
            if metadata['channels']['eeg']['occipital']:
                for canal_O in  metadata['channels']['eeg']['occipital']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            # PRUEBA 2 - W
        
            if metadata['channels']['eeg']['frontal']:
                for canal_O in  metadata['channels']['eeg']['frontal']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            if metadata['channels']['eeg']['central']:
                for canal_O in  metadata['channels']['eeg']['central']:
                    Porcentaje_de_densidad = periodograma.loc['Alpha', canal_O].loc[epoca]    # podria evaluar la transicion de la etapa del despertar al sueño 
                    if(Porcentaje_de_densidad>0.5) :
                        print('Suma puntos para ser Wake')
                        Puntos[0] += 1
            if Puntos[0] == 0 :   # si no esta en ningun canal el ritmo alpha superior se suma puntos apra ser N1  -- Descarte N1 
                Puntos[1] = +1
                print('suma puntos para ser N1')

            
            """
            3. Score epochs without visually discernible alpha rhythm as stage W if ANY of the following are
            present:
            a. Eye blinks at a frequency of 0.5-2 Hz
            b. Reading eye movements
            c. Irregular, conjugate rapid eye movements associated wit normal or high chin muscle tone Slow eye movements (SEM): Conjugate, reasonably regular, sinusoidal eye movements with an initial deflection
            usually lasting >500 msec.   
            """
            """
            if len(eventos_en_rango_REM)!=0  & duracion_delEvento > estiamacion  & pendiente_de_las_ondas > estiamcion :  # tengo que considerar que en W se detectan movimientos de ojos conjugados, pero no son d emanera rapida
                print('Suma puntos para ser W') 
            if  # detectar amplitud alta del EMG 
            """
            #STAGE N1
            """

            In subjects who generate alpha rhythm, score stage N1 if the alpha rhythm is attenuated and
            replaced by low-amplitude, mixed-frequency activity for more than 50% of the
            epoch.N1,N2,N3
            """
            dendidad_espectral = []
            band_names = [ 'Theta', 'Alpha', 'Sigma'] # 'Delta', 'Beta' saco estas 2 pq supongo que en la frecuencia mixta no participan
            varianza_x_canales = []
            
            for canal_eeg in metadata['channels']['eeg']:
                if  metadata['channels']['eeg'][canal_eeg]:
                    for canal in  metadata['channels']['eeg'][canal_eeg]:
                        for banda in band_names:
                            Porcentaje_de_densidad = periodograma.loc[banda, canal][epoca] 
                            dendidad_espectral.append(Porcentaje_de_densidad)
                    varianza = np.var(dendidad_espectral)
                    varianza_x_canales.append(varianza)
            umbral_varianza = 0.1  # tipo por tirar ...  no sabria que umbral poner
            if np.var(varianza_x_canales) < umbral_varianza :
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if Nueva_anotacion[epoca-1] == 0: # osea indica que es wake la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if Nueva_anotacion[epoca-1] == 1: # osea indica que es N1 la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            if Nueva_anotacion[epoca-1] == 4 :#osea indica que es  rem la epoca anterior 
                print('suma puntos para ser N1')
                Puntos[1] += 1
            """
            3. In subjects who do not generate alpha rhythm, score stage N1 commencing with the earliest of
            ANY of the following phenomena:N1,N2,N3,N4
            a. EEG activity in range of 4-7 Hz with slowing of background frequencies by ≥1 Hz from those of stage W
            b. Vertex sharp waves
            c. Slow eye movements
            """
            # STAGE 2
            """
            2. Begin scoring stage N2 (in absence of criteria for N3) if EITHER OR BOTH of the following occur
            during the first half of that epoch or the last half of the previous epoch:N1,N2,N3,N4
            a. One or more K complexes unassociated with arousals
            b. One or more trains of sleep spindles
            """
            
            if len(eventos_en_rango_spindle) != 0 or (len(eventos_en_rango_sw)!= 0 and np.sum(duracion__en_rango_sw) < 0.2*30):  # se considera que no se N3 para evalaur N2
                print('Suma puntos para ser N2')  
                                                                                            # se pone 0.2*30 para descartar que sea N3 dado que muchos kc juntos correspondena  etapa n3`
                Puntos[2] += 100 # osea si ocurre un complejo K o un spindle  se la considera Huso del sueño, por lo que se suma 100 puntos
                
            else :
                Puntos[1] += 1  # por descarte elijo a N1
                print('suma puntos para ser N1')
                
                """
                Continue to score epochs with low-amplitude, mixed-frequency EEG activity without K
                complexes or sleep spindles as stage N2 if they are preceded by epochs containing EITHER of the
                following:
                a. K complexes unassociated with arousals
                b. Sleep spindles
                """ 
            if len(eventos_en_rango_EpocaAnterior_spindle)!= 0 or len(eventos_en_rango_EpocaAnterior_sw)!= 0 :
                print('suma puntos para ser N2')
                Puntos[2] += 1
            if Nueva_anotacion[epoca-1] == 1 :# si la epoca anterior era N1 suma putos para que esta sea N2  - Consultar
                print('suma puntos para ser N2')
                Puntos[2] += 1
            if Nueva_anotacion[epoca-1] ==2: # osea indica que es N2 la epoca anterior - Consultar
                print('suma puntos para ser N2')
                Puntos[2] += 1

            # STAGE 3
            """
            Score stage N3 when ≥20% of an epoch consists of slow wave activity, irrespective of
            age.N2,N3,N4
            """
            
            if  np.sum(duracion__en_rango_sw) > 0.2*30 :
                print('suma puntos para ser N3')
                Puntos[3] += 100  # en la proxima ponerle 100 y..
            
            else :
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            if Nueva_anotacion[epoca-1] ==3: # osea indica que es N3 la epoca anterior   -Consultar
                print('suma puntos para ser N3')
                Puntos[3] += 1
            if Nueva_anotacion[epoca-1] ==2: # osea indica que es N2 la epoca anterior - Consultar
                print('suma puntos para ser N3')
                Puntos[3] += 1
            
            
            """
            Note 4. In stage N3, the chin EMG is of variable amplitude, often lower than in stage N2 sleep and sometimes as low
            as in stage R sleep.
            """
            # STAGE R
        
            """
            2. Score stage R sleep in epochs with ALL of the following phenomena:N1,N2,N3
            a. Low-amplitude, mixed-frequency EEG
            b. Low chin EMG tone
            c. Rapid eye movements
            """
            if len(eventos_en_rango_REM)!=0 :
                print('suma puntos para ser rem')
                Puntos[4] += 1

            varianza_x_canales = []    ##################### revisar igualq  mismo con n1
            for canal_eeg in metadata['channels']['eeg']:
                if  metadata['channels']['eeg'][canal_eeg]:
                    for canal in  metadata['channels']['eeg'][canal_eeg]:
                        for banda in band_names:
                            Porcentaje_de_densidad = periodograma.loc[banda, canal][epoca] 
                            dendidad_espectral.append(Porcentaje_de_densidad)
                    varianza = np.var(dendidad_espectral)
                    umbral_varianza = 0.1  # tipo por tirar ...  no sabria que umbral poner
                    if   varianza < umbral_varianza :
                            print('suma puntos para ser r')
                            Puntos[4] += 1
            """
            3. Continue to score stage R sleep, even in the absence of rapid eye movements, for epochs
            following one or more epochs of stage R as defined in rule I.2 above, IF the EEG continues to
            show low-amplitude, mixed-frequency activity without K complexes or sleep spindles AND the
            chin EMG tone remains low for the majority of the epoch.N4
            """
            if Nueva_anotacion[epoca-1] == 4 and (len(eventos_en_rango_spindle) != 0 or (len(eventos_en_rango_sw)!= 0)): # osea indica que  no hay usos del sueño ni spindle y al epoca anterior era rem
                print('suma puntos para ser rem')
                Puntos[4] += 1
            if Nueva_anotacion[epoca-1] == 1 : # osea indico que hay fase 3
                print('suma puntos para ser rem')
                Puntos[4] += 1
            
            elif Puntos[4] == 0 :
                Puntos[1] += 1
                print('suma puntos para ser N1')

            
            # creo esta nueva etapa  dependiendo del puntaje
            if np.std(Puntos) == 0 :
                epoca_elejida = 1 # osea si todas las anotaciones tienen la misma probabilidad se elije a N1 como candidata
            else :
                epoca_elejida = np.argmax(Puntos)
            Nueva_anotacion.append(epoca_elejida)
     
    
    return Nueva_anotacion
def Final_Vote_No_coincidencia(results, raw):
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
    # genero un canal stim y agrego los eventos
    pre_stim =np.zeros_like(np.array(np.arange(raw.n_times))) #[:,:][0][0]
    count = 0
    for index, value in enumerate(Candidatos_a_revision):
        if value == 1 :
            pre_stim[index*30*int(raw.info['sfreq'])]  = 1  
    stim_chan = pre_stim.reshape(1,-1)

    mask_info = mne.create_info(ch_names=["STIM_SAMPLE_TO_EVALUATE"],
                                sfreq=raw.info["sfreq"],
                                ch_types=["stim"]
                            )
    raw_mask = mne.io.RawArray(data=stim_chan,
                            info=mask_info,
                            first_samp=raw.first_samp
                            )
    raw.add_channels([raw_mask], force_update_info=True)

    return raw,Candidatos_a_revision
def Final_Vote_UmbralGSSC(results, raw):
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
            incertidumbre.append(1)
        else:
            incertidumbre.append(0)

    Candidatos_a_revision = incertidumbre

  # genero un canal stim y agrego los eventos
    pre_stim =np.zeros_like(np.array(np.arange(raw.n_times))) #[:,:][0][0]
    count = 0
    for index, value in enumerate(Candidatos_a_revision):
        if value == 1 :
            pre_stim[index*30*int(raw.info['sfreq'])]  = 1  
    stim_chan = pre_stim.reshape(1,-1)

    mask_info = mne.create_info(ch_names=["STIM_SAMPLE_TO_EVALUATE"],
                                sfreq=raw.info["sfreq"],
                                ch_types=["stim"]
                            )
    raw_mask = mne.io.RawArray(data=stim_chan,
                            info=mask_info,
                            first_samp=raw.first_samp
                            )
    raw.add_channels([raw_mask], force_update_info=True)

    return raw,Candidatos_a_revision
def Final_Vote(results, raw):
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
    
    if sum(No_concidencias) > len(anotaciones_gssc)*0.25 :  # en estas lineas de codigo se implementa la estrategia de usar los pesos de gssc
        # para epocas candidatas a revision, solo se hara uso de esta estrategia cuando la implemntacin de las no coincidencias
        # de un numero mayor al 0.25 del raw completo del estudio
        
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
                incertidumbre.append(1)
            else:
                incertidumbre.append(0)

        Candidatos_a_revision = incertidumbre

  # genero un canal stim y agrego los eventos
    pre_stim =np.zeros_like(np.array(np.arange(raw.n_times))) #[:,:][0][0]
    count = 0
    for index, value in enumerate(Candidatos_a_revision):
        if value == 1 :
            pre_stim[index*30*int(raw.info['sfreq'])]  = 1  
    stim_chan = pre_stim.reshape(1,-1)

    mask_info = mne.create_info(ch_names=["STIM_SAMPLE_TO_EVALUATE"],
                                sfreq=raw.info["sfreq"],
                                ch_types=["stim"]
                            )
    raw_mask = mne.io.RawArray(data=stim_chan,
                            info=mask_info,
                            first_samp=raw.first_samp
                            )
    raw.add_channels([raw_mask], force_update_info=True)

    return raw
def classify_file(file_data, metadata, classifiers):
    """
    Args:
        file_data (_type_): _description_
        metadata (_type_): _description_
        classifiers (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    results = {}
    for clasiff   in classifiers :
        if clasiff == 'YASA':
            result = ClassifYASA(file_data, metadata)
        if clasiff == 'GSSC' :
            result = ClassifGSSC(file_data, metadata)
        results[clasiff] = result
    return results
    """
    results = {}

    for clasiff   in classifiers :
        if clasiff == 'YASA':
            result = ClassifYASA(file_data, metadata)
        if clasiff == 'GSSC' :
            result = ClassifGSSC(file_data, metadata)
        results[clasiff] = result
    return results
def sleep_stage_classification_file(file_data, metadata, results, classifiers = None):
    """_summary_

    Args:
        file_data (mne.raw): archivo raw de mne que contiene la señal de polisomnografia del estudio
        metadata (dict):diccionario con variables necesarias para hacer la clasiicacion
        tiene que tener el siguiente formato :
        dict = {
        channels :{ 'eog':[], 'eeg':{'frontal':[], 'central':[], 'parietal':[]},'emg': []} 
        }
        classifiers (list): lista con los nombres de los clasificadores a usar si se agrega debera ser como minimo de 2 clasificadores. 
        Por defecto los valores son 'YASA', 'GSSC'        
        

    Returns:
        array: retorna un array con los valores de la clasificacion final
    """
  
    if  classifiers == None :
        classifiers = ['YASA', 'GSSC']
    ### 1ra parte del pipeline
   
    #results = classify_file(file_data, metadata, classifiers) # prueba pipeline 1,2,3,4   # PARA LA EVALAUCION 14 DEJO COMENTADO ESTA LINEA PORQUE RELEEO LOS RESULTADOS DE LAS EVALAUCION 12 
    
    #raw = Final_Vote(results,file_data)  # devuelve el archivo raw con un canal de stim donde se indica cuales eventos Fueron detectados con poca presicion en ambois modelos de acuerdoa  algun criterio
    #raw, Candidatos_a_revision  =Final_Vote_No_coincidencia(results,file_data) # prueba pipeline 3 y 10  # EN LA PRUEBA 12 EN ADELANTE NO USO CANDIDATO A REVISION   
    #raw, Candidatos_a_revision  = Final_Vote_UmbralGSSC(results,file_data) # prueba pipeline 4 y prueba 7
    ### 2da parte del pipeline
    
    #prediccion = AASM_Rules(file_data,metadata, results)  # prueba pipeline 1,2,3,4
    #prediccion = AASM_rules_todos(file_data, metadata) Prueba pipeline 5 (SIN FINAL VOTE)
    #prediccion =  AASM_rules_todos_conGSSC(file_data,metadata, results) # prueba pipeline 6 (SIN FINAL VOTE)

    #prediccion = AASM_RulesDirecto(raw, metadata, results) # Prueba 7 y 10
    #prediccion = AASM_RulesDirecto_todos_conGSSC(raw, metadata, results) # prueba 8, 9 me con fundi al realizar la 9 y termino siendo igual a la 8, hago prueba 11 corrigo el error q teniaantes en la 8y9 del porcentaje en tiempo de sw y de %de alpha en W
    #prediccion = ASSM_RulesDirectConReevaluacion(raw, metadata, results)  # prueba 12 (el cod de la prueba 13 cambio asiq meti la prueba 12 en otra funcion ) y 13  no necesita candidatoa  rev
    prediccion= AASM_RulesPrueba19(file_data, metadata, results)
    return prediccion
