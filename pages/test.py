import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premium Dashboard")

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
# INPUT TABLE (MAIN LAYOUT)
# ----------------------------
input_df = pd.DataFrame({
    "Month": months.strftime("%b-%Y"),
    "ZQ Price": [94.50] * 12
})

st.subheader("Enter ZQ Prices (Rolling 12 Months)")

edited_df = st.data_editor(
    input_df,
    num_rows="fixed",
    use_container_width=True
)

edited_df["Implied Month Rate"] = 100 - edited_df["ZQ Price"]

# ----------------------------
# BUILD MONTH STRUCTURE
# ----------------------------
rows = []

for i, m in enumerate(months):
    end = m + pd.offsets.MonthEnd(1)
    days = (end - m).days + 1

    meeting_series = fomc_dates[
        (fomc_dates.month == m.month) &
        (fomc_dates.year == m.year)
    ]

    meeting_date = (
        meeting_series.iloc[0]
        if not meeting_series.empty
        else pd.NaT
    )

    if pd.notna(meeting_date):
        pre = (meeting_date - m).days
        post = days - pre
    else:
        pre, post = days, 0

    rows.append({
        "Month": m.strftime("%b-%Y"),
        "MonthStart": m,
        "Meeting Date": meeting_date,
        "Days": days,
        "Pre": pre,
        "Post": post,
        "Month Rate": edited_df.iloc[i]["Implied Month Rate"]
    })

df = pd.DataFrame(rows)

# ----------------------------
# AUTO ANCHOR (LAST PRE-MEETING ZQ)
# ----------------------------
meeting_rows = df[df["Meeting Date"].notna()]

if meeting_rows.empty:
    st.error("No FOMC meetings found in the next 12 months.")
    st.stop()

first_meeting = meeting_rows.iloc[0]["Meeting Date"]

anchor_candidates = df[
    (df["Meeting Date"].isna()) &
    (df["MonthStart"] < first_meeting)
]

if anchor_candidates.empty:
    st.error(
        "First contract contains a meeting. "
        "Add one pre-meeting ZQ month."
    )
    st.stop()

anchor = anchor_candidates.iloc[-1]["Month Rate"]

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
# DISPLAY OUTPUT
# ----------------------------
st.caption(
    f"As of {date.today()} | "
    f"Anchor rate: {anchor:.2f}% | "
    f"Total priced change: {df['Meeting Premium (bps)'].sum():.1f} bps"
)

st.subheader("Meeting-by-Meeting Decomposition")

st.dataframe(
    df[[
        "Month",
        "Meeting Date",
        "Meeting Premium (bps)",
        "Policy Path"
    ]],
    use_container_width=True
)

st.subheader("Meeting Premiums (bps)")
st.bar_chart(df.set_index("Month")["Meeting Premium (bps)"])
