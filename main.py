import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==============================================================================
# Securely fetch secrets from GitHub Actions environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1399391666")
LOOKBACK = 30

# ==============================================================================
# TOOL 1: DATA FETCHER
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
# MODEL ARCHITECTURE
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
# TOOL 2: VOLATILITY INFERENCE ENGINE
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
# TOOL 3: AUTOMATED RISK DECISION ENGINE
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
# TOOL 4: TELEGRAM ALERT DISPATCHER
# ==============================================================================
def send_telegram_alert(report_text, bot_token, chat_id):
    if not bot_token or not chat_id:
        print("[Tool 4 - Alert Dispatcher]: Error - Missing Telegram Bot Token or Chat ID.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": report_text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("[Tool 4 - Alert Dispatcher]: Telegram notification sent successfully!")
        else:
            print(f"[Tool 4 - Alert Dispatcher]: Failed to send. Error: {response.text}")
    except Exception as e:
        print(f"[Tool 4 - Alert Dispatcher]: Network error: {e}")

# ==============================================================================
# WORKFLOW ORCHESTRATOR
# ==============================================================================
def main():
    print("=== EXECUTING AUTONOMOUS MARKET RISK AGENT WORKFLOW ===")

    # 1. Fetch Fresh Data
    df = get_market_data()
    current_gvz = df['Gold_Vol'].iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    print(f"[Tool 1 - Data Fetcher]: Data loaded up to {latest_date}.")

    # 2. Preprocess & Scale Data
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

    # 3. Train PyTorch GRU Network
    model = GoldVolatilityGRU()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("Training PyTorch GRU Network...")
    epochs = 60
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 4. Run Model Inference
    pred_gvz, pred_scaled = predict_next_day_volatility(df, model, scaler)
    print(f"[Tool 2 - Neural Model]: Prediction calculated (Scaled: {pred_scaled:.4f}).")

    # 5. Apply Policy Decision Rules
    risk_status, recommendation, diff = evaluate_risk_level(pred_gvz, current_gvz)
    print(f"[Tool 3 - Risk Engine]: Policy rules applied.")

    # 6. Dispatch Telegram Alert
    message = (
        f"🚨 *AUTONOMOUS GOLD RISK REPORT* 🚨\n\n"
        f"📅 *Date:* `{latest_date}`\n"
        f"📊 *Current Gold Vol (^GVZ):* `{current_gvz:.2f}`\n"
        f"🔮 *Predicted 24h Vol:* `{pred_gvz:.2f}`\n"
        f"📈 *Forecasted Shift:* `{'+' if diff >= 0 else ''}{diff:.2f}` pts\n\n"
        f"⚠️ *Risk Level:* *{risk_status}*\n"
        f"💡 *Action:* {recommendation}"
    )
    send_telegram_alert(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

if __name__ == "__main__":
    main()
