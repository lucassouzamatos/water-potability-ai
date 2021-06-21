import pandas as pd
from normalizer import Normalizer 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

class Network:
    def __init__(self) -> None:
      self.__setup__()

    def __setup__(self) -> None:
      self.normalizer = Normalizer()
      self.__set_data_train__()   

    '''
      set 80% of dataset for train
    '''
    def __set_data_train__(self):
      dataset_train_potable = self.normalizer.dataset_potable.loc[self.normalizer.dataset_potable.index < 1022]
      dataset_train_unpotable = self.normalizer.dataset_unpotable.loc[self.normalizer.dataset_unpotable.index < 1598]
      self.dataset_train = pd.concat([dataset_train_potable, dataset_train_unpotable])

    def best_algorithm(self):
      models= {
        'MLPClassifier': MLPClassifier(
          hidden_layer_sizes=7,
          learning_rate_init=0.65, 
          max_iter=700, 
          random_state=105
        ),
        'RandomForestClassifier': RandomForestClassifier(
          n_estimators=250,
          max_depth=500,
          max_leaf_nodes=500,
          random_state=105
        )
      }

      x_train = self.dataset_train.loc[:, [ 
        'ph', 
        'Hardness', 
        'Solids', 
        'Chloramines', 
        'Sulfate', 
        'Conductivity', 
        'Organic_carbon', 
        'Trihalomethanes',
        'Turbidity'
      ]]
      y_train = self.dataset_train['Potability']

      dataset_test_potable = self.normalizer.dataset_potable.loc[self.normalizer.dataset_potable.index >= 1022]
      dataset_test_unpotable = self.normalizer.dataset_unpotable.loc[self.normalizer.dataset_unpotable.index >= 1598]

      dataset_test = pd.concat([dataset_test_potable, dataset_test_unpotable])
      x_test = dataset_test.loc[:, [
        'ph', 
        'Hardness', 
        'Solids', 
        'Chloramines', 
        'Sulfate', 
        'Conductivity', 
        'Organic_carbon', 
        'Trihalomethanes',
        'Turbidity'
      ]]
      y_test = dataset_test['Potability']
      
      training_scores = []
      testing_scores = []
      
      for key, value in models.items():
          value.fit(x_train, y_train)
          train_score= value.score(x_train,  y_train)
          test_score= value.score(x_test, y_test)
          training_scores.append(train_score)
          testing_scores.append(test_score)
          
          print(f"{key}\n")
          print(f"Score of testing: {test_score} \n")

if __name__ == '__main__':
  network = Network()
  network.best_algorithm()