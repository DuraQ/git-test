#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import keras
import pandas as pdimport tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing
from matplotlib import pyplot as plt


# In[7]:


df = pd.read_csv(r"G:\ADS_Projects\kaggle\kaggle.csv")
dates = pd.to_datetime(df["Time"]) 


# In[8]:


# split rows we're looking to predict
pred = df[df['Series 3'].isna()]
pred.drop(['Series 3'], axis = 1, inplace= True)
pred.drop(['Time'], axis = 1, inplace= True)
# keep rows we'll build model on
df = df.dropna()
y = df['Series 3']
df.drop(['Series 3'], axis = 1, inplace= True)
df.drop(['Time'], axis = 1, inplace= True)


# #### Try taking a random 80/20 split and see how that works. This will have to be expanded to put proper CV in place

# In[4]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


# In[5]:


train_features = train_dataset.copy()
test_features = test_dataset.copy()
pred_features = pred.copy()

train_labels = y[train_dataset.index]
test_labels = y.drop(train_dataset.index)


# #### Normalize the training data to see if this gives better results

# In[60]:



normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))


# In[62]:


print(normalizer.mean.numpy())


# In[63]:


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


# In[64]:


kaggle = np.array(train_features['Series 1'])

kaggle_normalizer = preprocessing.Normalization(input_shape=[1,])
kaggle_normalizer.adapt(kaggle)


# In[65]:


kaggle_model = tf.keras.Sequential([
    kaggle_normalizer,
    layers.Dense(units=1)
])

kaggle_model.summary()


# In[66]:


kaggle_model.predict(kaggle[:10])


# In[67]:


kaggle_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')


# In[68]:


get_ipython().run_cell_magic('time', '', "history = kaggle_model.fit(\n    train_features['Series 1'], train_labels,\n    epochs=200,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.2)")


# In[69]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[70]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)


# In[71]:


plot_loss(history)


# In[72]:


test_results = {}

test_results['kaggle_model'] = kaggle_model.evaluate(
    test_features['Series 1'],
    test_labels, verbose=0)


# In[73]:


x = tf.linspace(0.0, 250, 251)
y = kaggle_model.predict(x)


# In[74]:


def plot_kaggle(x, y):
  plt.scatter(train_features['Series 1'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Series 1')
  plt.ylabel('Series 3')
  plt.legend()


# In[75]:


plot_kaggle(x,y)


# In[76]:


linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])


# In[77]:


linear_model.predict(train_features[:10])


# In[78]:


linear_model.layers[1].kernel


# In[79]:


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')


# In[80]:


get_ipython().run_cell_magic('time', '', 'history = linear_model.fit(\n    train_features, train_labels, \n    epochs=100,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.2)')


# In[81]:


plot_loss(history)


# In[82]:


test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


# In[83]:


def build_and_compile_model(norm):
  model = tf.keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


# In[84]:


dnn_kaggle_model = build_and_compile_model(kaggle_normalizer)


# In[85]:


get_ipython().run_cell_magic('time', '', "history = dnn_kaggle_model.fit(\n    train_features['Series 1'], train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)")


# In[86]:


plot_loss(history)


# In[87]:


x = tf.linspace(0.0, 250, 251)
y = dnn_kaggle_model.predict(x)


# In[88]:


plot_kaggle(x, y)


# In[89]:


test_results['dnn_kaggle_model'] = dnn_kaggle_model.evaluate(
    test_features['Series 1'], test_labels,
    verbose=0)


# In[90]:


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


# In[91]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    train_features, train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)')


# In[92]:


plot_loss(history)


# In[93]:


test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)


# In[94]:


pd.DataFrame(test_results, index=['mean_squared_error Series.3']).T


# In[95]:


test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[97]:


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


# In[98]:


dnn_model.save('dnn_model')


# In[99]:


reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)


# In[100]:


pd.DataFrame(test_results, index=['mean_squared_error Series.3']).T


# In[101]:


final_predictions = dnn_model.predict(pred_features).flatten()


# In[102]:


final_predictions


# In[105]:


np.savetxt('C:/Temp/ddn_pred.csv', final_predictions, delimiter=",")


# In[120]:


pred_csv = pd.read_csv('C:/Temp/ddn_pred.csv', header=None)
final_df = pd.read_csv(r"G:\ADS_Projects\kaggle\kaggle.csv")
final_pred_frame = final_df[final_df['Series 3'].isna()]
final_pred_frame = final_pred_frame.reset_index()


# In[121]:


final_pred_frame['Series 3'] = pred_csv.squeeze()
final_pred_frame['Series 1'] = pd.Series(["{0:.4f}".format(val) for val in final_pred_frame['Series 1']], index = final_pred_frame.index)
final_pred_frame['Series 2'] = pd.Series(["{0:.4f}".format(val) for val in final_pred_frame['Series 2']], index = final_pred_frame.index)
final_pred_frame['Series 3'] = pd.Series(["{0:.4f}".format(val) for val in final_pred_frame['Series 3']], index = final_pred_frame.index)


# In[122]:


final_pred_frame.drop('index', axis = 1, inplace= True)
final_pred_frame.to_csv(r"G:\ADS_Projects\kaggle\prediction_25092020.csv", index = False)


# In[ ]:




