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
 
  def add_labels(df, zero_class='F2', one_class='F3'):
    
    df['label'] = -99
    df['label'][df.diagnosis_code.str.contains(zero_class)] = 0
    df['label'][df.diagnosis_code.str.contains(one_class)] = 1
    
    return df   
    
  
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
    
    return df_sparse_matrix
   
  
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
  
  
  def split_to_train_test(sequences, labels, test_frac=0.20, valid_frac=0.25):
    train_data, test_data, train_label, test_label = train_test_split(sequences, labels, test_size=test_frac, shuffle=True)
    train_data, valid_data, train_label, valid_label = train_test_split(sequences, labels, test_size=valid_frac, shuffle=True) #0.8x0.25=0.2
    return train_data, train_label, test_data, test_label, valid_data, valid_label
  
  def save_files(export_path, train_data, train_label, test_data, test_label, valid_data, valid_label):
    np.save(os.path.join(export_path, 'train_data.npy'),  np.array(train_data))
    np.save(os.path.join(export_path, 'valid_data.npy'),  np.array(valid_data))
    np.save(os.path.join(export_path, 'test_data.npy'),   np.array(test_data))
    np.save(os.path.join(export_path, 'train_label.npy'), np.array(train_label))
    np.save(os.path.join(export_path, 'valid_label.npy'), np.array(valid_label))
    np.save(os.path.join(export_path, 'test_label.npy'),  np.array(test_label))
    
  @classmethod
  def run(cls, df):
    df_ = cls.add_labels(df, zero_class='F2', one_class='F3')
    df_matrix_ = cls.sparse_matrix(df_)
    df_matrix_padded = cls.matrix_to_padded_sequence(df_matrix_)
    sequences, labels= cls.build_data_files(df_matrix_padded, df)    
    train_data, train_label, test_data, test_label, valid_data, valid_label = cls.split_to_train_test(sequences, labels, test_frac=0.20, valid_frac=0.25)
    cls.save_files(export_path, train_data, train_label, test_data, test_label, valid_data, valid_label)
  
  if __name__ == "__main__":
    Preprocessor.run(df)
  
  
