import sys
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import uuid
from Analysis.utils import slice_data, concat_data
from RF_Classifier.utils import transform_stats
from sklearn.metrics import f1_score


# Command Line: $ python predict.py blabla/path/test
test_path = sys.argv[1]  # 'blabla/path/test'
model_path = 'RF_Data/Random.pkl'  # our best model
columns = ["HR", "O2Sat", "Temp", "MAP", "Resp", "AST", "BUN",
           "Alkalinephos", "Calcium", "Creatinine", "Glucose", "Bilirubin_total",
           "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "ICULOS"]

# Prepare data:
u = str(uuid.uuid4())
os.mkdir(u)
os.mkdir(u+'/sliced')
labels = slice_data(src_dir=test_path, dst_dir=u+'/sliced')
shutil.move(u+'/sliced/labels.csv', u+'/labels.csv')
df = concat_data(src_dir=u+'/sliced', dst_dir=u, name='test', impute='rf')
tdf = transform_stats(df, columns)
transformed_df = tdf.copy()
transformed_df['SepsisLabel'] = transformed_df['PatientID'].apply(lambda k: int(labels.loc[k]))

# Split columns
X = np.array(transformed_df.drop(columns=['PatientID', 'SepsisLabel']))
y = np.array(transformed_df['SepsisLabel'])
names = np.array(transformed_df['PatientID'])

# Scale data using the fitted (train) scaler:
# scaler = joblib.load('RF_Data/scaler.gz')
# X = scaler.transform(X)

# Load our selected model and predict:
model = joblib.load(model_path)
predictions = model.predict(X)


# Write output CSV:
df = pd.DataFrame(predictions.astype(int), index=names, columns=None)
df.to_csv('prediction.csv', header=False)

# Print score:
print("F1 score:\t{}".format(f1_score(y, predictions)))
