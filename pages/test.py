import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premium Dashboard")

# ======================================================
# 1. AUTO ROLL NEXT 12 CALENDAR MONTHS
# ======================================================
today = pd.Timestamp.today()

months = pd.date_range(
    today + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

# ======================================================
# 2. FOMC MEETING CALENDAR (EDIT ONCE PER YEAR)
# ======================================================
fomc_dates = pd.to_datetime([
    "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-11-05", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28",
    "2027-06-16", "2027-07-28"
])

# ======================================================
# 3. MAIN INPUT TABLE (USER ENTERS PRICES)
# ======================================================
st.subheader("Enter Rolling 12 ZQ Prices")

input_df = pd.DataFrame({
    "Month": months.strftime("%b-%Y"),
    "ZQ Price": [94.50] * 12
})

edited_df = st.data_editor(
    input_df,
    num_rows="fixed",
    use_container_width=True
)

edited_df["Implied Month Rate"] = 100 - edited_df["ZQ Price"]

# ======================================================
# 4. BUILD MONTH / MEETING STRUCTURE
# ======================================================
rows = []

for i, m in enumerate(months):
    month_end = m + pd.offsets.MonthEnd(1)
    days_in_month = (month_end - m).days + 1

    # IMPORTANT: fomc_dates is a DatetimeIndex → NO .iloc
    meeting_idx = fomc_dates[
        (fomc_dates.month == m.month) &
        (fomc_dates.year == m.year)
    ]

    meeting_date = meeting_idx[0] if len(meeting_idx) > 0 else pd.NaT

    if pd.notna(meeting_date):
        pre_days = (meeting_date - m).days
        post_days = days_in_month - pre_days
    else:
        pre_days = days_in_month
        post_days = 0

    rows.append({
        "Month": m.strftime("%b-%Y"),
        "MonthStart": m,
        "Meeting Date": meeting_date,
        "Days": days_in_month,
        "Pre": pre_days,
        "Post": post_days,
        "Month Rate": edited_df.iloc[i]["Implied Month Rate"]
    })

df = pd.DataFrame(rows)

# ======================================================
# 5. AUTO ANCHOR (LAST PRE-MEETING ZQ)
# ======================================================
meeting_rows = df[df["Meeting Date"].notna()]

if meeting_rows.empty:
    st.error("No FOMC meetings found in the next 12 months.")
    st.stop()

first_meeting_date = meeting_rows.iloc[0]["Meeting Date"]

anchor_candidates = df[
    (df["Meeting Date"].isna()) &
    (df["MonthStart"] < first_meeting_date)
]

if anchor_candidates.empty:
    st.error(
        "First ZQ month contains a meeting.\n"
        "Add one fully pre-meeting ZQ contract."
    )
    st.stop()

anchor_rate = anchor_candidates.iloc[-1]["Month Rate"]

# ======================================================
# 6. SOLVE MEETING-BY-MEETING PREMIUMS
# ======================================================
prev_rate = anchor_rate
meeting_moves = []

for _, r in df.iterrows():
    if r["Post"] == 0:
        meeting_moves.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new_rate = (r["Month Rate"] - w_pre * prev_rate) / w_post
    meeting_moves.append((new_rate - prev_rate) * 100)  # bps
    prev_rate = new_rate

df["Meeting Premium (bps)"] = meeting_moves
df["Policy Path"] = anchor_rate + np.cumsum(df["Meeting Premium (bps)"]) / 100

# ======================================================
# 7. OUTPUT
# ======================================================
st.caption(
    f"As of {date.today()} | "
    f"Anchor rate: {anchor_rate:.2f}% | "
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
