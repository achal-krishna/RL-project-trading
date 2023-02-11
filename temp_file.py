import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('BTC.csv')
df['Date']=pd.to_datetime(df['Date'])
df['Close']=df['Price'].str.replace(',','').astype(float)
df['Open']=df['Open'].str.replace(',','').astype(float)
df['High']=df['High'].str.replace(',','').astype(float)
df['Low']=df['Low'].str.replace(',','').astype(float)
df.pop('Change %')
df.pop('Vol.')
df.pop('Price')
#df['Change %']=df['Change %'].str.replace('%','').astype(float)
#df["Vol."]=df["Vol."].replace({"K":"*1e3", "M":"*1e6", "B":"*1e9"}, regex=True).map(pd.eval).astype(int)
df.set_index('Date',inplace=True)
df.head()

env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
env=DummyVecEnv([env_maker])

model=A2C('MlpPolicy',env,verbose=1)
model.learn(total_timesteps=10000)