import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premia → SR3 Pricing & Spreads")

# ======================================================
# 1. FOMC MEETING CALENDAR
# ======================================================
fomc_dates = pd.to_datetime([
    "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-11-05", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28",
    "2027-06-16", "2027-07-28"
]).sort_values()

# ======================================================
# 2. FIND FIRST FUTURE MEETING → ANCHOR MONTH
# ======================================================
today = pd.Timestamp.today().normalize()
future_meetings = fomc_dates[fomc_dates >= today]

if future_meetings.empty:
    st.error("No future FOMC meetings found.")
    st.stop()

first_meeting = future_meetings.min()

# Anchor = month strictly BEFORE first meeting month
anchor_month = (
    first_meeting.to_period("M").to_timestamp()
    - pd.DateOffset(months=1)
)

months = pd.date_range(anchor_month, periods=13, freq="MS")

# ======================================================
# 3. ZQ INPUT TABLE (ANCHOR + 12 MONTHS)
# ======================================================
st.subheader("ZQ Prices (Anchor + Next 12 Months)")

zq_df = pd.DataFrame({
    "Month": months.strftime("%b-%Y"),
    "ZQ Price": [94.50] * 13
})

zq_df = st.data_editor(
    zq_df,
    num_rows="fixed",
    use_container_width=True
)

zq_df["Month Rate"] = 100 - zq_df["ZQ Price"]

# ======================================================
# 4. BUILD MONTH / MEETING STRUCTURE
# ======================================================
rows = []

for i, m in enumerate(months):
    month_end = m + pd.offsets.MonthEnd(1)
    days = (month_end - m).days + 1

    meeting_idx = fomc_dates[
        (fomc_dates.month == m.month) &
        (fomc_dates.year == m.year)
    ]

    meeting_date = meeting_idx[0] if len(meeting_idx) > 0 else pd.NaT

    if pd.notna(meeting_date):
        pre = (meeting_date - m).days
        post = days - pre
    else:
        pre, post = days, 0

    rows.append({
        "Month": m.strftime("%b-%Y"),
        "Meeting Date": meeting_date,
        "Days": days,
        "Pre": pre,
        "Post": post,
        "Month Rate": zq_df.iloc[i]["Month Rate"]
    })

month_df = pd.DataFrame(rows)

# ======================================================
# 5. ZQ → MEETING PREMIA
# ======================================================
anchor_rate = month_df.iloc[0]["Month Rate"]
anchor_label = month_df.iloc[0]["Month"]

work_df = month_df.iloc[1:].reset_index(drop=True)

prev_rate = anchor_rate
meeting_premia = []

for _, r in work_df.iterrows():
    if r["Post"] == 0:
        meeting_premia.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new_rate = (r["Month Rate"] - w_pre * prev_rate) / w_post
    meeting_premia.append((new_rate - prev_rate) * 100)
    prev_rate = new_rate

work_df["Meeting Premium (bps)"] = meeting_premia
work_df["Policy Path"] = anchor_rate + np.cumsum(work_df["Meeting Premium (bps)"]) / 100

st.subheader("Implied FOMC Meeting Premia (from ZQ)")
st.dataframe(
    work_df[["Month", "Meeting Date", "Meeting Premium (bps)", "Policy Path"]],
    use_container_width=True
)

# ======================================================
# 6. SR3 QUARTERS (ORDERED)
# ======================================================
sr3_quarters = pd.DataFrame([
    ("SR3H6", "2026-03-18", "2026-06-17"),
    ("SR3M6", "2026-06-17", "2026-09-16"),
    ("SR3U6", "2026-09-16", "2026-12-16"),
    ("SR3Z6", "2026-12-16", "2027-03-17"),
], columns=["SR3", "Start", "End"])

sr3_quarters["Start"] = pd.to_datetime(sr3_quarters["Start"])
sr3_quarters["End"] = pd.to_datetime(sr3_quarters["End"])
sr3_quarters = sr3_quarters.sort_values("Start").reset_index(drop=True)

# ======================================================
# 7. MAP MEETING PREMIA → SR3 (CORRECT BASE RATE)
# ======================================================
sr3_summary_rows = []

for _, q in sr3_quarters.iterrows():
    total_days = (q["End"] - q["Start"]).days

    # --- Base policy rate at SR3 start ---
    base_rate = anchor_rate
    for _, m in work_df.iterrows():
        meet = m["Meeting Date"]
        if pd.isna(meet):
            continue
        if meet < q["Start"]:
            base_rate += m["Meeting Premium (bps)"] / 100
        else:
            break

    # --- Add meetings inside the SR3 quarter ---
    sr3_rate = base_rate
    for _, m in work_df.iterrows():
        meet = m["Meeting Date"]
        if pd.isna(meet) or meet < q["Start"] or meet >= q["End"]:
            continue

        days_after = (q["End"] - meet).days
        weight = days_after / total_days
        sr3_rate += weight * (m["Meeting Premium (bps)"] / 100)

    sr3_summary_rows.append({
        "SR3": q["SR3"],
        "Start": q["Start"],
        "Implied SR3 Rate (%)": round(sr3_rate, 3),
        "Implied SR3 Price": round(100 - sr3_rate, 3),
    })

sr3_summary_df = (
    pd.DataFrame(sr3_summary_rows)
    .sort_values("Start")
    .reset_index(drop=True)
)

st.subheader("SR3 – Implied Rates & Prices")
st.dataframe(
    sr3_summary_df[["SR3", "Implied SR3 Rate (%)", "Implied SR3 Price"]],
    use_container_width=True
)

# ======================================================
# 8. ZQ PRICE SPREADS (ZQ1 - ZQ2)
# ======================================================
zq_spreads_df = pd.DataFrame([
    {
        "Spread": f"{zq_df.iloc[i]['Month']} - {zq_df.iloc[i+1]['Month']}",
        "ZQ Price Spread": round(
            zq_df.iloc[i]["ZQ Price"] - zq_df.iloc[i+1]["ZQ Price"], 4
        )
    }
    for i in range(len(zq_df) - 1)
])

st.subheader("ZQ Calendar Spreads (Price)")
st.dataframe(zq_spreads_df, use_container_width=True)

# ======================================================
# 9. SR3 PRICE SPREADS (FRONT - NEXT)
# ======================================================
sr3_spreads_df = pd.DataFrame([
    {
        "Spread": f"{sr3_summary_df.iloc[i]['SR3']} - {sr3_summary_df.iloc[i+1]['SR3']}",
        "SR3 Price Spread": round(
            sr3_summary_df.iloc[i]["Implied SR3 Price"]
            - sr3_summary_df.iloc[i+1]["Implied SR3 Price"], 4
        )
    }
    for i in range(len(sr3_summary_df) - 1)
])

st.subheader("SR3 Calendar Spreads (Price)")
st.dataframe(sr3_spreads_df, use_container_width=True)

# ======================================================
# 10. FOOTER
# ======================================================
st.caption(
    f"As of {date.today()} | "
    f"Anchor month: {anchor_label} | "
    f"Anchor rate: {anchor_rate:.2f}%"
)
