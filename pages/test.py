import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premium Dashboard")

# ----------------------------
# MANUAL ZQ INPUTS
# ----------------------------
st.sidebar.header("Manual ZQ Inputs")

zq_prices = []
for i in range(12):
    zq_prices.append(
        st.sidebar.number_input(
            f"ZQ Month {i+1} Price",
            value=94.50,
            step=0.005,
            key=f"zq_{i}"
        )
    )

prices = pd.DataFrame({
    "price": zq_prices
})
prices["implied_rate"] = 100 - prices["price"]

# ----------------------------
# AUTO MONTH ROLL
# ----------------------------
today = pd.Timestamp.today()

months = pd.date_range(
    today + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

# ----------------------------
# FOMC CALENDAR (EDIT ONCE PER YEAR)
# ----------------------------
fomc_dates = pd.to_datetime([
    "2026-03-18","2026-05-06","2026-06-17","2026-07-29",
    "2026-09-16","2026-11-05","2026-12-16",
    "2027-01-27","2027-03-17","2027-04-28",
    "2027-06-16","2027-07-28"
])

# ----------------------------
# BUILD MONTH STRUCTURE
# ----------------------------
rows = []

for i, m in enumerate(months):
    end = m + pd.offsets.MonthEnd(1)
    days = (end - m).days + 1

    meeting = fomc_dates[
        (fomc_dates.month == m.month) &
        (fomc_dates.year == m.year)
    ]
    meeting = meeting.iloc[0] if len(meeting) else None

    if meeting is not None:
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
# AUTO ANCHOR (LAST PRE-MEETING ZQ)
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
    moves.append((new - prev) * 100)  # bps
    prev = new

df["Meeting Premium (bps)"] = moves
df["Policy Path"] = anchor + np.cumsum(df["Meeting Premium (bps)"]) / 100

# ----------------------------
# DISPLAY
# ----------------------------
st.caption(
    f"As of {date.today()} | "
    f"Anchor: {anchor:.2f}% | "
    f"Total priced change: {df['Meeting Premium (bps)'].sum():.1f} bps"
)

st.dataframe(
    df[["Month","Meeting Date","Meeting Premium (bps)","Policy Path"]],
    use_container_width=True
)

st.bar_chart(df.set_index("Month")["Meeting Premium (bps)"])
