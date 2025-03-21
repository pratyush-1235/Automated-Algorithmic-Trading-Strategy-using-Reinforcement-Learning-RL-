# 📈 AI-Powered Algorithmic Trading using Deep Reinforcement Learning

A Streamlit-based interactive application that simulates an **AI-driven trading strategy** on Google stock data (GOOGL) using **Deep Q-Learning (DQN)**. This project integrates **technical indicators** with **deep reinforcement learning** to visualize buy/sell decisions on stock price data.

---

## 🚀 Features

- ✅ Google Stock Price Data (last 3 months)
- ✅ Technical Indicators:
  - SMA 10 (Simple Moving Average)
  - SMA 50 (Simple Moving Average)
  - RSI (Relative Strength Index)
- ✅ Custom OpenAI Gym Trading Environment
- ✅ Deep Q-Learning Agent (DQN)
- ✅ Streamlit UI for visualization
- ✅ Buy/Sell Signal Plot with Matplotlib

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎈
- PyTorch ⚡
- OpenAI Gym
- Yahoo Finance API (yfinance)
- Matplotlib 📊
- Pandas / NumPy

---

## 🧠 How It Works

1. **Stock Data** is fetched from Yahoo Finance API (last 3 months).
2. A **custom trading environment** is built using OpenAI Gym.
3. A **Deep Q-Learning (DQN) model** is trained to make trading decisions.
4. The model learns an optimal policy to Buy, Sell, or Hold based on rewards.
5. The strategy is visualized with Buy (green arrow) and Sell (red arrow) points on the stock price chart.

---

## 📸 Screenshot

> Example chart showing AI-predicted Buy/Sell points on Google stock.

![Trading Strategy Visualization](Screenshot%202025-03-21%20120421.png)

---

## ⚙️ Installation & Usage

1. **Install Required Libraries**

```bash
pip install streamlit yfinance gym torch matplotlib pandas numpy
