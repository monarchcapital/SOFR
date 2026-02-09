import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premium Dashboard")

# ======================================================
# 1. AUTO ROLL 13 CALENDAR MONTHS (ANCHOR + 12)
# ======================================================
today = pd.Timestamp.today()

months = pd.date_range(
    today,                # anchor month INCLUDED
    periods=13,
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
# 3. MAIN INPUT TABLE (13 ZQ PRICES)
# ======================================================
st.subheader("Enter ZQ Prices (Anchor + Next 12 Months)")

input_df = pd.DataFrame({
    "Month": months.strftime("%b-%Y"),
    "ZQ Price": [94.50] * 13
})

edited_df = st.data_editor(
    input_df,
    num_rows="fixed",
    use_container_width=True
)

edited_df["Month Rate"] = 100 - edited_df["ZQ Price"]

# ======================================================
# 4. BUILD MONTH / MEETING STRUCTURE
# ======================================================
rows = []

for i, m in enumerate(months):
    month_end = m + pd.offsets.MonthEnd(1)
    days_in_month = (month_end - m).days + 1

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
        "Month Rate": edited_df.iloc[i]["Month Rate"]
    })

df = pd.DataFrame(rows)

# ======================================================
# 5. ANCHOR (FIRST ROW, GUARANTEED PRE-MEETING)
# ======================================================
anchor_rate = df.iloc[0]["Month Rate"]

df_work = df.iloc[1:].reset_index(drop=True)

# ======================================================
# 6. SOLVE MEETING PREMIUMS
# ======================================================
prev_rate = anchor_rate
meeting_moves = []

for _, r in df_work.iterrows():
    if r["Post"] == 0:
        meeting_moves.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new_rate = (r["Month Rate"] - w_pre * prev_rate) / w_post
    meeting_moves.append((new_rate - prev_rate) * 100)  # bps
    prev_rate = new_rate

df_work["Meeting Premium (bps)"] = meeting_moves
df_work["Policy Path"] = anchor_rate + np.cumsum(df_work["Meeting Premium (bps)"]) / 100

# ======================================================
# 7. OUTPUT
# ======================================================
st.caption(
    f"As of {date.today()} | "
    f"Anchor month: {df.iloc[0]['Month']} | "
    f"Anchor rate: {anchor_rate:.2f}% | "
    f"Total priced change: {df_work['Meeting Premium (bps)'].sum():.1f} bps"
)

st.subheader("Meeting-by-Meeting Decomposition")

st.dataframe(
    df_work[[
        "Month",
        "Meeting Date",
        "Meeting Premium (bps)",
        "Policy Path"
    ]],
    use_container_width=True
)

st.subheader("Meeting Premiums (bps)")
st.bar_chart(df_work.set_index("Month")["Meeting Premium (bps)"])
