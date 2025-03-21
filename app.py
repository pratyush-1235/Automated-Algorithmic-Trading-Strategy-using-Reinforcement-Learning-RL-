import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# 🎯 Streamlit UI
st.title("📈 AI-Powered Algorithmic Trading using Deep Reinforcement Learning")

# ✅ Fetch New Dataset (Google Stock)
@st.cache_data
def get_stock_data():
    df = yf.download("GOOGL", period="3mo", interval="1d")  # Reduced time for faster processing
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df.dropna(inplace=True)
    return df[['Close', 'SMA_10', 'SMA_50', 'RSI']].values

data = get_stock_data()
st.write(f"Loaded {len(data)} days of Google stock price data.")

# ✅ Custom Trading Environment (with Indicators)
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000
        self.action_space = gym.spaces.Discrete(3)  # 0 = Buy, 1 = Sell, 2 = Hold
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return np.array(self.data[-1]), 0, True, {}

        self.current_step += 1
        price = self.data[self.current_step][0]
        reward = 0

        if action == 0:  # Buy
            self.balance -= price
            reward = random.uniform(-0.5, 1)  # Reduced randomness for faster learning
        elif action == 1:  # Sell
            self.balance += price
            reward = random.uniform(-0.5, 1)

        done = self.current_step >= len(self.data) - 1
        return np.array(self.data[self.current_step]), reward, done, {}

    def reset(self):
        self.current_step = 0
        return np.array(self.data[self.current_step])

# ✅ Deep Q-Learning Model (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),  # Reduced neurons for speed
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ✅ Train DQN Agent (Faster)
def train_agent(env, episodes=50):  # Reduced episodes for faster output
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # Increased learning rate
    criterion = nn.MSELoss()
    memory = deque(maxlen=500)
    epsilon = 1.0
    gamma = 0.9  # Faster learning

    for episode in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, torch.tensor(next_state, dtype=torch.float32), done))

            if len(memory) > 16:  # Smaller batch size for speed
                batch = random.sample(memory, 16)
                for s, a, r, s_next, d in batch:
                    target = r + (gamma * torch.max(model(s_next)).item() * (1 - int(d)))
                    output = model(s)[a]
                    loss = criterion(output, torch.tensor(target))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            state = torch.tensor(next_state, dtype=torch.float32)

        epsilon = max(0.1, epsilon * 0.98)

    return model

env = TradingEnv(data)
model = train_agent(env)

# ✅ Run Trading Strategy & Visualize
state = torch.tensor(env.reset(), dtype=torch.float32)
done = False
prices, actions = [], []

while not done:
    with torch.no_grad():
        action = torch.argmax(model(state)).item()

    next_state, _, done, _ = env.step(action)
    prices.append(next_state[0])
    actions.append(action)
    state = torch.tensor(next_state, dtype=torch.float32)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(prices, label="Stock Price", color="blue")
buy_signals = [i for i, x in enumerate(actions) if x == 0]
sell_signals = [i for i, x in enumerate(actions) if x == 1]

ax.scatter(buy_signals, np.array(prices)[buy_signals], color="green", marker="^", label="Buy Signal")
ax.scatter(sell_signals, np.array(prices)[sell_signals], color="red", marker="v", label="Sell Signal")

ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.legend()
ax.set_title("Deep RL Trading Strategy Visualization for GOOGL")

st.pyplot(fig)
st.success("Trading strategy execution complete! 🚀")
