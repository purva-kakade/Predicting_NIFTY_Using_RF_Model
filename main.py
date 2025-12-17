
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score
import yfinance as yf

nifty = yf.Ticker("^NSEI").history(period="max")
if "Dividends" in nifty.columns:
    del nifty["Dividends"]
if "Stock Splits" in nifty.columns:
    del nifty["Stock Splits"]
nifty["Tomorrow"] = nifty["Close"].shift(-1)
nifty["Target"] = (nifty["Tomorrow"] > nifty["Close"]).astype(int)
nifty = nifty.loc["1990-01-01":].copy()

sg = yf.Ticker("^STI").history(period="max")[["Close"]].rename(columns={"Close": "SG_Close"})
sg["SG_pct"] = sg["SG_Close"].pct_change()
sg = sg.reindex(nifty.index).ffill().fillna(0.0)

sp500 = yf.Ticker("^GSPC").history(period="max")[["Close"]].rename(columns={"Close": "SP500_Close"})
sp500["SP500_pct"] = sp500["SP500_Close"].pct_change()
sp500 = sp500.reindex(nifty.index).ffill().fillna(0.0)

nifty = nifty.merge(sp500[["SP500_pct"]], left_index=True, right_index=True, how="left")
nifty = nifty.merge(sg[["SG_pct"]], left_index=True, right_index=True, how="left")

model = RandomForestClassifier(n_estimators=1000, min_samples_split=50, random_state=1, class_weight="balanced")

def predict(train, test, predictors, model, threshold=0.6):
    model.fit(train[predictors], train["Target"])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    pred_series = pd.Series(preds, index=test.index, name="Predictions")
    prob_series = pd.Series(probs, index=test.index, name="Prob_Pos")
    combined = pd.concat([test["Target"], pred_series, prob_series], axis=1)
    return combined

def BackTest(data, model, predictors, start=2500, setp=250):
    all_predictions = []
    for i in range(start, data.shape[0], setp):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i + setp].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = nifty.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    nifty[ratio_column] = nifty["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    nifty[trend_column] = nifty.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

nifty["SG_roll_5"] = nifty["SG_pct"].rolling(5).sum()
nifty["SP500_roll_5"] = nifty["SP500_pct"].rolling(5).sum()
new_predictors += ["SG_pct", "SG_roll_5", "SP500_pct", "SP500_roll_5"]

nifty = nifty.dropna()

predictions = BackTest(nifty, model, new_predictors)

predictions = predictions.dropna()
print(predictions["Predictions"].value_counts())
print(predictions["Target"].value_counts() / predictions.shape[0])
print(precision_score(predictions["Target"], predictions["Predictions"]))
