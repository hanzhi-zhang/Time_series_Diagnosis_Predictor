import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_file_path = ''
export_path = ''
df = pd.read_csv(input_file_path)
df['document_date'] = pd.to_datetime(df['document_date'])

class Preprocessor(object):
  
  def build_data_files(padded, df):
    sequences = []
    labels = []
    
    for idx, seq in enumerate(padded):
      sequences.append(seq[:, 1:])
      
      if seq[:,0][0] == 0:
        labels.append([1,0])
      elif seq[:,0][0] == 1:
        labels.append([0,1])
      else:
        continue
    
    return sequences, labels
   
   
    
  def matrix_to_padded_sequence(df_sparse_matrix):
    g = df_sparse_matrix.reset_index().sort_values('document_date').groupby('id')
    all_patient_records = []
    
    for g_name, group in g:
      cols_to_drop = ['index', 'id', 'document_date']
      patient_record_history = group.drop(columns=cols_to_drop).to_numpy()
      all_patient_records.append(patient_record_history)
    padded = pad_sequences(all_patient_records, dtype = 'int32', value = -99, padding = 'post')
    patient0_n_notes = all_patient_records[0].shape[0]
    if np.array_equal(padded[0][:patient0_n_notes], all_patient_records[0]):
      return padded
    else:
      print("Error! Cannot convert matrix to padded sequence.")
      
  
  def sparse_matrix(df):
    df['dummy'] = 1
    df_sparse_matrix = pd.pivot_table(df, values='dummy',
                                      index = ['id', 'document_date'],
                                      columns = ['drug_name'],
                                      aggfunc = np.max,
                                      fill_value = 0)
    df_sparse_matrix = (df[['id','label']]
                        .drop_duplicates()
                        .set_index('id')
                        .join(df_sparse_matrix.reset_index().set_index('id'))
                       )
    df_sparse_matrix = (df_sparse_matrix
                        .reset_index()
                        .sort_values(['id','document_date'])
                       )
    
    def calc_delta_time(df_sparse_matrix):
      timediff = (df_sparse_matrix
                  .groupby('id')['document_date']
                  .transform(lambda x:x.diff().dt.days)
                  .fillna(0)
                  .astype('int16')
                 )
      return timediff
    
    timediff = calc_delta_time(df_sparse_matrix)
    
    df_sparse_matrix.inset(3, 'timediff', timediff)
  
  def add_labels(df, zero_class='F2', one_class='F3'):
    df['label'] = -99
    df['label'][df.diagnosis_code.str.contains(zero_class)] = 0
    df['label'][df.diagnosis_code.str.contains(one_class)] = 1
    return df

  
  
