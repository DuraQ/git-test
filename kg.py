#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[3]:


import datetime
from sklearn.metrics import mean_absolute_error
import pandas as pd
import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True)


# In[4]:


pdf = pd.read_csv('./data/df_kaggle_train.csv', usecols=['ValDt','Time','Net_Pos'])
pdf.loc[-1] = ['2018-02-27', '17:23:00', 191.27875] # mean of neighbors
pdf.loc[-2] = ['2018-05-29', '17:35:00', 220.02465] # mean of neighbors
pdf.sort_values(by=['ValDt', 'Time'], inplace = True)
pdf['index_date'] = pd.to_datetime(pdf['ValDt'] + ' ' + pdf['Time'])
date_time = pd.to_datetime(pdf['ValDt'] + ' ' + pdf['Time'])
#pdf.drop(['ValDt','Time'], axis=1, inplace=True)
#dropping Date and index fields
pdf.set_index(pdf['index_date'], inplace=True)
pdf.drop(['ValDt','Time','index_date'], axis=1, inplace=True)
#pdf.drop(['Time','index_date'], axis=1, inplace=True)
df=pdf


# In[5]:


#timestamp_s = df.index.map(datetime.datetime.timestamp)


# In[6]:


#seconds = 24*60*60


# In[7]:


#df['sec_sin'] = np.sin(timestamp_s * (2 * np.pi / seconds))
#df['sec_cos'] = np.sin(timestamp_s * (2 * np.pi / seconds))


# In[8]:


column_indices = {name: i for i, name in enumerate(df.columns)}
column_indices


# In[9]:


n = len(df)
n


# In[9]:


#str(df.index[round(n*.7)].date())


# In[10]:


#df.index[round(n*.7)].date()


# In[11]:


#final_train = str(df.index[round(n*.7)].date())
#final_test = str(df.index[round(n*.9)].date())


# In[12]:


#final_train = df.index[df.ValDt == final_train][-1]
#final_test = df.index[df.ValDt == final_test][-1]


# In[13]:


#df[:final_test-690]


# In[10]:


# index 371066 corresponds to 18:30 at 2019-04-03
# index 388342 corresponds to 18:30 at 2019-09-30
train_df = df[0:371067]
val_df = df[0:388342]
test_df = df[0:]
test_df.shape


# In[11]:


num_features = df.shape[1]
num_features


# In[12]:


#train_mean = train_df.mean()
#train_std = train_df.std()

#train_df = (train_df - train_mean) / train_std
#val_df = (val_df - train_mean) / train_std
#test_df = (test_df - train_mean) / train_std


# In[13]:


prediction_periods = 10
y_hats=[]
for i in reversed(range(prediction_periods)):
    h= i + 1
    window_index = (len(df) - h)
    y_win = df[: window_index].tail(10)
    y_hats.append(y_win.mean())


# In[14]:


mean_absolute_error(df.tail(prediction_periods), y_hats)


# In[15]:


#input_width = start index of each day - 541 (number of minutes to go back to 9:30
#sliding window start should be start_index - 541 - 120*691 - (691 - 541))

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        #store the raw data
        #print(train_df)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        #work out label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_column_indices = {name:i for i,name in
                                         enumerate(label_columns)}
        self.column_indices = {name:i for i,name in
                               enumerate(train_df.columns)}

        #work our the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        #print(self.total_window_size)
        self.input_slice = slice(0, input_width)
        #self.input_slice = slice(0, input_width - 541)
        #self.input_slice = slice(input_width - 82920, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        #print(features)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        #print(data)
        data = np.array(data, dtype=np.float32)
        #print(self.total_window_size)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=691,
            shuffle=False,
            batch_size=32,)      

        ds = ds.map(self.split_window)

        return ds
    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


# In[16]:


#This class is used just to create the validation window of 691

class ValWindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 val_df,
                 label_columns=None):
        #store the raw data
        #print(train_df)
        #self.train_df = train_df
        self.val_df = val_df
        #self.test_df = test_df

        #work out label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_column_indices = {name:i for i,name in
                                         enumerate(label_columns)}
        self.column_indices = {name:i for i,name in
                               enumerate(train_df.columns)}

        #work our the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        #print(self.total_window_size)
        self.input_slice = slice(0, input_width)
        #self.input_slice = slice(0, input_width - 541)
        #self.input_slice = slice(input_width - 82920, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        #print(features)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        #print(data)
        data = np.array(data, dtype=np.float32)
        #print(self.total_window_size)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=691,
            shuffle=False,
            batch_size=32,)      

        ds = ds.map(self.split_window)

        return ds
    
    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)


# In[17]:


val_window = ValWindowGenerator(
    #input_width = 691, label_width= 691, shift=0,
    input_width = 691, label_width= 691, shift=0,
    val_df=val_df,
    label_columns=['Net_Pos']
)


# In[18]:


single_step_window = WindowGenerator(
    input_width = 82380, label_width= 691, shift=1231,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['Net_Pos']
)


# In[19]:


single_step_window.train


# In[20]:


single_step_window.val


# In[21]:


val_window.val


# In[22]:


class Baseline(tf.keras.Model):
    def __init__(self, label_index = None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        #print("inputs")
        #print(inputs.numpy())
        #print(inputs.shape)
        #print("results")
        #print(result.numpy())
        #print(result.shape)
        return result[:,:, tf.newaxis]


# In[23]:


baseline = Baseline(label_index = column_indices['Net_Pos'])


# In[24]:


baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])


# In[25]:


val_performance = {}


# In[26]:


val_window.val


# In[27]:


#simply predict the value at the the same time on previous day
val_performance['Baseline'] = baseline.evaluate(val_window.val)
val_performance


# In[28]:


df[round(df.Net_Pos ,5)== 74.93223]


# In[29]:


#baseline.save('tfmodel')


# In[30]:


linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1)
])


# In[31]:


linear.compile(loss=tf.losses.MeanSquaredError(),
               optimizer=tf.optimizers.Adam(),
               metrics = [tf.metrics.MeanAbsoluteError()])


# In[32]:


df[round(df.Net_Pos ,5)== 135.86287]


# In[33]:


#linear.fit(single_step_window.train, epochs=3, validation_data=val_window.val)
linear.fit(single_step_window.train, epochs=20, validation_data=single_step_window.train)


# In[34]:


val_performance['Linear'] = linear.evaluate(single_step_window.val)
val_performance


# In[35]:


df[round(df.Net_Pos ,5)== 135.86287]


# In[36]:


#linear.save('tfmodel')


# In[37]:


MAX_EPOCHS = 4


# In[38]:


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss = tf.losses.MeanSquaredError(),
                  optimizer= tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    model.fit(window.train,
              epochs=MAX_EPOCHS,
              validation_data=window.val,
              callbacks=[early_stopping])


# In[39]:


#compile_and_fit(linear, single_step_window)
compile_and_fit(linear, single_step_window)


# In[40]:


# Dense model

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

compile_and_fit(dense, single_step_window)
#val_performance['Dense'] = dense.evaluate(single_step_window.val)
val_performance['Dense'] = dense.evaluate(val_window.val)
val_performance


# In[41]:


dense.save('tfmodel')


# In[47]:


CONV_WIDTH = 3
#conv_window = WindowGenerator(
#    input_width = CONV_WIDTH, label_width= 1, shift=1,
#    train_df=train_df, val_df=val_df, test_df=test_df,
#    label_columns=['Net_Pos']
#)

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=[CONV_WIDTH,],
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

compile_and_fit(conv_model, single_step_window)
val_performance['Conv'] = conv_model.evaluate(val_window.val)
val_performance


# In[ ]:


lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])

#wide_window = WindowGenerator(
#    input_width=82920, label_width=691, shift=691,
#    train_df=train_df, val_df=val_df, test_df=test_df,
#    label_columns=['Net_Pos']
#)

#compile_and_fit(lstm_model, wide_window)
#val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
compile_and_fit(lstm_model, single_step_window)
val_performance['LSTM'] = lstm_model.evaluate(val_window.val)
val_performance


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class TestDataGenerator():
    def __init__(self, input_width,
                 testing_df):
        #store the raw data
        #self.train_df = train_df
        #self.val_df = val_df
        self.testing_df = testing_df

        #work out label column indices
        #self.label_columns = label_columns
        #if label_columns is not None:
        #    self.label_column_indices = {name:i for i,name in
        #                                 enumerate(label_columns)}
        self.column_indices = {name:i for i,name in
                               enumerate(testing_df.columns)}

        #work our the window parameters
        self.input_width = self.testing_df.shape[1]
        #self.label_width = label_width
        #self.shift = shift
        #self.total_window_size = input_width + shift
        self.input_slice = slice(0, self.input_width)
        #self.input_slice = slice(input_width - 82920, input_width)
        #self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        #self.label_start = self.total_window_size - self.label_width
        #self.labels_slice = slice(self.label_start, None)
        #self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        #labels = features[:, self.labels_slice, :]
        #if self.label_columns is not None:
        #    labels = tf.stack(
        #        [labels[:,:, self.column_indices[name]] for name in
        #         self.label_columns],
        #        axis=-1)

        inputs.set_shape([None, self.input_width, None])
        #labels.set_shape([None, self.label_width, None])

        return inputs#, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width,
            sequence_stride=1,
            shuffle=False,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    #@property
    #def train(self):
    #    return self.make_dataset(self.train_df)

    #@property
    #def val(self):
    #    return self.make_dataset(self.val_df)

    @property
    def testing(self):
        return self.make_dataset(self.testing_df)


# In[ ]:





# In[ ]:





# In[ ]:


pdf = pd.read_csv('./data/df_kaggle_train.csv', usecols=['ValDt','Time','Net_Pos'])
pdf.loc[-1] = ['2018-02-27', '17:23:00', 191.27875] # mean of neighbors
pdf.loc[-2] = ['2018-05-29', '17:35:00', 220.02465] # mean of neighbors
pdf.sort_values(by=['ValDt', 'Time'], inplace = True)
#pdf['recent_available'] = pdf['Net_Pos'].shift(540)
#pdf['previous_day'] = pdf['Net_Pos'].shift(1382)
pdf['ValDt'] = pd.to_datetime(pdf['ValDt'])
pdf['Time'] = pd.to_datetime(pdf['Time'])
pdf.drop(['ValDt','Time'], axis=1,inplace=True)


# In[ ]:


#Worked Example


# In[ ]:


pdf


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




