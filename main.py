import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import telebot

# ==============================================================================
# CONFIGURATION & TELEGRAM INITIALIZATION
# ==============================================================================
# Reads token from environment variable or falls back to your string token
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN") or "8859986286:AAHNXLoesx0HWAcxgv5ie2UoN5YT7v9Lih8"
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
LOOKBACK = 30

# ==============================================================================
# TOOL 1: MARKET DATA FETCHER
# ==============================================================================
def get_market_data(start_date="2015-01-01"):
    tickers = ['GLD', '^GVZ', 'UUP', '^TNX']
    df = yf.download(tickers, start=start_date)['Close'].dropna()

    df['Gold_Return'] = np.log(df['GLD'] / df['GLD'].shift(1))
    df['Gold_Vol'] = df['^GVZ']
    df['USD_Index'] = df['UUP']
    df['Yield_10Y'] = df['^TNX']
    return df[['Gold_Return', 'Gold_Vol', 'USD_Index', 'Yield_10Y']].dropna()

# ==============================================================================
# NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class GoldVolatilityGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# ==============================================================================
# TOOL 2: INFERENCE ENGINE
# ==============================================================================
def predict_next_day_volatility(raw_data, trained_model, scaler):
    latest_30_days = raw_data.values[-LOOKBACK:]
    scaled_window = scaler.transform(latest_30_days)
    input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)

    trained_model.eval()
    with torch.no_grad():
        scaled_pred = trained_model(input_tensor).item()

    dummy = np.zeros((1, 4))
    dummy[0, 1] = scaled_pred
    unscaled_pred = scaler.inverse_transform(dummy)[0, 1]

    return unscaled_pred, scaled_pred

# ==============================================================================
# TOOL 3: RISK ASSESSMENT ENGINE
# ==============================================================================
def evaluate_risk_level(predicted_gvz, current_gvz):
    diff = predicted_gvz - current_gvz

    if predicted_gvz >= 22.0 or diff > 2.0:
        risk_status = "HIGH RISK / VOLATILITY SPIKE WARNING"
        recommendation = "Reduce long position size; hedge portfolio using gold put options."
    elif predicted_gvz >= 16.0 or diff > 0.5:
        risk_status = "ELEVATED RISK"
        recommendation = "Monitor order book closely; tighten stop-loss thresholds."
    else:
        risk_status = "NORMAL / LOW RISK"
        recommendation = "Maintain standard position allocations."

    return risk_status, recommendation, diff

# ==============================================================================
# WORKFLOW PIPELINE
# ==============================================================================
def generate_risk_report():
    df = get_market_data()
    current_gvz = df['Gold_Vol'].iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(len(scaled_data) - LOOKBACK):
        X.append(scaled_data[i : i + LOOKBACK])
        y.append(scaled_data[i + LOOKBACK, 1])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.8)
    X_train = torch.tensor(X[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)

    model = GoldVolatilityGRU()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 60
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    pred_gvz, pred_scaled = predict_next_day_volatility(df, model, scaler)
    risk_status, recommendation, diff = evaluate_risk_level(pred_gvz, current_gvz)

    return (
        f"🚨 *AUTONOMOUS GOLD RISK REPORT* 🚨\n\n"
        f"📅 *Date:* `{latest_date}`\n"
        f"📊 *Current Gold Vol (^GVZ):* `{current_gvz:.2f}`\n"
        f"🔮 *Predicted 24h Vol:* `{pred_gvz:.2f}`\n"
        f"📈 *Forecasted Shift:* `{'+' if diff >= 0 else ''}{diff:.2f}` pts\n\n"
        f"⚠️ *Risk Level:* *{risk_status}*\n"
        f"💡 *Action:* {recommendation}"
    )

# ==============================================================================
# TELEGRAM COMMAND HANDLERS
# ==============================================================================
@bot.message_handler(commands=['start', 'report'])
def handle_start(message):
    status_msg = bot.reply_to(message, "⏳ Fetching live market data & running model... Please wait.")
    try:
        report = generate_risk_report()
        bot.send_message(message.chat.id, report, parse_mode="Markdown")
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error generating report: {str(e)}")
    finally:
        try:
            bot.delete_message(message.chat.id, status_msg.message_id)
        except Exception:
            pass

if __name__ == "__main__":
    print("Bot is listening 24/7 for /start commands...")
    bot.infinity_polling()
