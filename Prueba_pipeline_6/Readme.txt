La evaluación es sobre 15 sujetos de cada dataset, se evaluaron todas las épocas sin criterio de selección 
asi es la saldia del sleep_stae_classification :
def sleep_stage_classification_file(file_data, metadata, classifiers = None):
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
   
    results = classify_file(file_data, metadata, classifiers) # prueba pipeline 1,2,3,4
    
    #raw = Final_Vote(results,file_data)  # devuelve el archivo raw con un canal de stim donde se indica cuales eventos 
                                           #fueron detectados con poca presicion en ambois modelos de acuerdoa  algun criterio
    #raw, Candidatos_a_revision  =Final_Vote_No_coincidencia(results,file_data) # prueba pipeline 3
    #raw, Candidatos_a_revision  = Final_Vote_UmbralGSSC(results,file_data) # prueba pipeline 4
    ### 2da parte del pipeline
    
    #prediccion = AASM_Rules(file_data,metadata, results)  # prueba pipeline 1,2,3,4
    #prediccion = AASM_rules_todos(file_data, metadata) Prueba pipeline 5 (SIN FINAL VOTE)
    prediccion =  AASM_rules_todos_conGSSC(file_data,metadata, results) # prueba pipeline 6 (SIN FINAL VOTE)


    return prediccion, results['GSSC'], results['YASA']