import numpy as np
import pandas as pd
import seaborn as sns
import mne_bids 
import mne
import os 
import yasa
import pandas as pd
from KC_algorithm.model import score_KCs
from KC_algorithm.plotting import KC_from_probas
from scipy.signal import welch
##############################################################
# la idea es generar funciones que tengan todas las mismas salidas y en lo posible la mismas entradas 
##############################################################

# detectar eventos spindle  tengo que seguir trabajando en este codigo

def SpindleDetect(raw, canal): # parametros : definir ...
    """_summary_

    Args:
        raw (_type_): _description_
        canales (_type_): _description_

    Returns:
        _type_: _description_
    """
    # seleciono los canales que voy a evaluar spindle
    #raw.pick_types(include = canal)  #  voy a elejir que use un solo canal para hacer esta etapa
   
    # hago la deteccion de spindle
    sp = yasa.spindles_detect(raw)
    #canal_mas_detect = max(sp.get_coincidence_matrix(scaled=False)) # aca elijo el canal que ams spindle detecto
   
    # selecciono el o los canales que quiero obtener e spindle
    peak = sp.summary() 
    canal_usado = peak[peak['Channel'] == canal[0]]
   
    
    # genero la matriz de con los eventos
    data_stim = (canal_usado['Peak']*raw.info['sfreq']).astype(int)
    pre_stim = np.zeros_like(np.array(np.arange(raw.n_times))) 
    pre_stim[data_stim] = 1
    stim_chan = pre_stim.reshape(1,-1)


    mask_info = mne.create_info(ch_names=["STIMspindle"],
                            sfreq=raw.info["sfreq"],
                            ch_types=["stim"]
                           )
    raw_mask = mne.io.RawArray(data=stim_chan,
                           info=mask_info,
                           first_samp=raw.first_samp
                          )
    raw.add_channels([raw_mask], force_update_info=True)
 
    return raw 
 # data stim tiene indica enq ue meustra ocurrio el spindle y raw tieene agregado el canal de estimulo
    # return stim_data # puede retornar el stim o una combinacion de de array cada uno para cada canal
"""
ALGORITMO kCOMPLEX
*Pasos para obtener llas anotaciones de compeljos K desde un edf raw*
 * Cargar EDF
 * Cargar anotaciones
 * Enviar el objeto a el RAW
 * enviar un objeto mne.Anottation con las anotaciones

 * genero el siguiente Dataframe 
       hypno = pd.DataFrame()
       hypno['onset'] = events_train[:,0]/raw.info['sfreq']
       hypno['dur'] = np.ones_like(events_train[:,0])*30
       hypno['label'] = events_train[:,2]
 * obtengo los eventos de Kcomplex 
 * aplico
      labels = np.where(probas>0.8,1,0)  principalemente onesets me indica en que momento ocurrio el Kcomplez y label que estadio del sueño pertenecia, oneset debe estar en meustras
      onsets = peaks[probas>0.5]   
"""

def kComplexDetect(raw, canal, dict, Annotacion_con__duracion  = True) :
    
    """
    raw type[ mne.raw] : data en crudo con los datos de las anotaciones seteados, es importante que las anotaciones esten seteadas en el raw y que la duracion de cazda evento sea de 30 segundos para cada anotacion.
    dict type[dic] : diccionario con el numero de etiqueta numerica que deseo que tenga el evento
    canal type[string] : string del canal seleccionado

    return archivo mne.raw seteado el canal de stim de este evento
    """
    if Annotacion_con__duracion :
        events_train, _ = mne.events_from_annotations(
            raw, event_id= dict, chunk_duration= 30.0)
        hypno = pd.DataFrame()
        hypno['onset'] = events_train[:,0]
        hypno['dur'] = [30 for _ in events_train[:,1]]
        hypno['label'] = events_train[:,2]
    else :
        events_train, _ = mne.events_from_annotations(
            raw, event_id= dict)
        hypno = pd.DataFrame()
        hypno['onset'] = events_train[:,0]/raw.info['sfreq']
        hypno['dur'] = np.ones_like(events_train[:,0])*30
        hypno['label'] = events_train[:,2]

    # clasifico los k complex
    wanted_channel = canal #
    CZ = np.asarray(
        [raw[count, :][0]for count, k in enumerate(raw.info['ch_names']) if
            k == wanted_channel]).ravel()*-1

    Fs = raw.info['sfreq']

    peaks, stage_peaks, d, probas = score_KCs(CZ, Fs, hypno,sleep_stages=list(dict.values()))  # list(dict.values()) me permite evaluar posibles complejos K en cualquier etapa evaluada
   
    # me quedo con  aquellos k complex que tienen mas del 80% de probabilidad
    onsets_ = peaks[probas>0.8]  # indica el numero de muesstra donde se encuentra elcomplejo k
    
    # genero un canal stim y agrego los eventos
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
    
    return  onsets_ , raw

# Revisar implementacion SW y REMdetect

def RemDetect(loc, roc, sf, raw) :   # los datos debene estar en uV
    """_summary_

    Args:
        loc (_type_): _description_
        roc (_type_): _description_
        sf (_type_): _description_
        raw (_type_): _description_

    Returns:
        _type_: _description_
    """
    rem = yasa.rem_detect(loc, roc, sf)

    # Get the detection dataframe
    events = rem.summary()

    
    data_stim = (events['Peak']*raw.info['sfreq']).astype(int)
    pre_stim = np.zeros_like(np.array(np.arange(raw.n_times))) 
    pre_stim[data_stim] = 1
    stim_chan = pre_stim.reshape(1,-1)
    
    mask_info = mne.create_info(ch_names=["STIMREM"],
                                sfreq=raw.info["sfreq"],
                                ch_types=["stim"]
                            )
    raw_mask = mne.io.RawArray(data=stim_chan,
                            info=mask_info,
                            first_samp=raw.first_samp
                            )
    raw.add_channels([raw_mask], force_update_info=True)

    return raw


def DetectorSW(raw):
    sw = yasa.sw_detect(raw)

    # Get the detection dataframe
    events = sw.summary()

    
    data_stim = (events['MidCrossing']*raw.info['sfreq']).astype(int)
    pre_stim = np.zeros_like(np.array(np.arange(raw.n_times))) 
    pre_stim[data_stim] = 1
    stim_chan = pre_stim.reshape(1,-1)
    
    mask_info = mne.create_info(ch_names=["STIMSW"],
                                sfreq=raw.info["sfreq"],
                                ch_types=["stim"]
                            )
    raw_mask = mne.io.RawArray(data=stim_chan,
                            info=mask_info,
                            first_samp=raw.first_samp
                            )
    raw.add_channels([raw_mask], force_update_info=True)

    return raw
    


################### -> revisar implementacion, echo
def Periodograma_Welch_por_segmento(raw, canales) :
    """
    
    _summary_

    Args:
        raw (_type_): _description_
        canales (_type_): _description_

    """

    # Create a 3-D array
    data = raw.get_data(canales)
    sf = raw.info['sfreq']

    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)

    #print(raw.ch_names)
    #print(data.shape, sf)   

    # calculo la ventana para calcular con PSD la densidad espectrar de potencia a los canales seleccionados
    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(data, sf, nperseg=win, axis=-1) 

    #freqs.shape, psd.shape



    # separo en bandas frecuenciales lo calculado 
    bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
        (12, 16, 'Sigma'), (16, 30, 'Beta')]

    # Calculate the bandpower on 3-D PSD array
    bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands)
    np.round(bandpower, 2)
    print(bandpower.shape)
    
    return  bandpower