import pandas as pd

class Normalizer:
  csv_data = 'dataset/water_potability.csv'  # file from work data

  def __init__(self) -> None:
    self.dataset = pd.read_csv(self.csv_data)
    self.__normalize_data__()
    self.__separate__()

  '''
    @ convert all info to number
  '''
  def __normalize_data__(self) -> None:
    self.dataset = self.dataset.apply(pd.to_numeric)
    self.dataset['ph'] = self.dataset['ph'].fillna(self.dataset.groupby('Potability')['ph'].transform('mean'))
    self.dataset['Sulfate'] = self.dataset['Sulfate'].fillna(self.dataset.groupby('Potability')['Sulfate'].transform('mean'))
    self.dataset['Trihalomethanes'] = self.dataset['Trihalomethanes'].fillna(self.dataset.groupby('Potability')['Trihalomethanes'].transform('mean'))
    
  '''
    separates the dataset where clause potable or unpotable
  '''
  def __separate__(self):
    self.dataset_potable = self.dataset.loc[self.dataset['Potability'] == 1]
    self.dataset_unpotable = self.dataset.loc[self.dataset['Potability'] == 0]
    self.dataset_potable = self.dataset_potable.reset_index()
    self.dataset_unpotable = self.dataset_unpotable.reset_index()

if __name__ == '__main__':
  normalizer = Normalizer()