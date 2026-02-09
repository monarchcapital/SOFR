import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premium Dashboard")

# ----------------------------
# LOAD LATEST PRICES
# ----------------------------
prices = pd.read_csv("data/zq_latest.csv")
asof = prices["asof"].iloc[0]

prices["implied_rate"] = 100 - prices["price"]

# ----------------------------
# DYNAMIC MONTHS
# ----------------------------
today = pd.Timestamp.today()
months = pd.date_range(
    today + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

# ----------------------------
# FOMC CALENDAR
# ----------------------------
fomc_dates = pd.to_datetime([
    "2026-03-18","2026-05-06","2026-06-17","2026-07-29",
    "2026-09-16","2026-11-05","2026-12-16",
    "2027-01-27","2027-03-17","2027-04-28",
    "2027-06-16","2027-07-28"
])

rows = []
for i, m in enumerate(months):
    end = m + pd.offsets.MonthEnd(1)
    days = (end - m).days + 1

    meeting = fomc_dates[(fomc_dates.month == m.month) &
                          (fomc_dates.year == m.year)]
    meeting = meeting.iloc[0] if len(meeting) else None

    if meeting:
        pre = (meeting - m).days
        post = days - pre
    else:
        pre, post = days, 0

    rows.append({
        "Month": m.strftime("%b-%Y"),
        "MonthStart": m,
        "Meeting Date": meeting,
        "Days": days,
        "Pre": pre,
        "Post": post,
        "Month Rate": prices.iloc[i]["implied_rate"]
    })

df = pd.DataFrame(rows)

# ----------------------------
# AUTO ANCHOR
# ----------------------------
first_meeting = df["Meeting Date"].dropna().iloc[0]

anchor = df[
    (df["Meeting Date"].isna()) &
    (df["MonthStart"] < first_meeting)
].iloc[-1]["Month Rate"]

# ----------------------------
# SOLVE MEETING PREMIUMS
# ----------------------------
prev = anchor
moves = []

for _, r in df.iterrows():
    if r["Post"] == 0:
        moves.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new = (r["Month Rate"] - w_pre * prev) / w_post
    moves.append((new - prev) * 100)
    prev = new

df["Meeting Premium (bps)"] = moves
df["Policy Path"] = anchor + np.cumsum(df["Meeting Premium (bps)"]) / 100

# ----------------------------
# DISPLAY
# ----------------------------
st.caption(f"As of {asof} | Anchor rate: {anchor:.2f}%")

st.dataframe(
    df[["Month","Meeting Date","Meeting Premium (bps)","Policy Path"]],
    use_container_width=True
)

st.bar_chart(df.set_index("Month")["Meeting Premium (bps)"])
