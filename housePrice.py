import os
import numpy as np
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit as stratify 
from sklearn.preprocessing import Imputer, LabelEncoder

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetchHousingData(housingUrl = HOUSING_URL, housingPath = HOUSING_PATH):
    if not os.path.isdir(housingPath):
        os.makedirs(housingPath)
    tgzPath = os.path.join(housingPath, "housing.tgz")
    urllib.request.urlretrieve(housingUrl, tgzPath)
    housingTgz = tarfile.open(tgzPath)
    housingTgz.extractall(path=housingPath)
    housingTgz.close()

def loadHousingData(housingPath = HOUSING_PATH):
    csvPath = os.path.join(housingPath, "housing.csv")
    return pd.read_csv(csvPath)

fetchHousingData()
housing = loadHousingData()

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace=True)

strata = stratify(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in strata.split(housing, housing["income_cat"]):
    stratTrainSet = housing.loc[train_index]
    stratTestSet = housing.loc[test_index]
for set in (stratTrainSet, stratTestSet):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = stratTrainSet.drop("median_house_value", axis=1)
housingLabels = stratTrainSet["median_house_value"].copy()

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
transformed = imputer.transform(housing_num)
housingTrans = pd.DataFrame(transformed, columns=housing_num.columns)    

encoder = LabelEncoder()
housingCat = housing["ocean_proximity"]    
housingCatEncoded = encoder.fit_transform(housingCat)
