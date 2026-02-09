import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premia → SR3 Pricing")

# ======================================================
# 1. FOMC MEETING CALENDAR (EDIT ONCE PER YEAR)
# ======================================================
fomc_dates = pd.to_datetime([
    "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-11-05", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28",
    "2027-06-16", "2027-07-28"
])

# ======================================================
# 2. FIND FIRST FUTURE MEETING → ANCHOR MONTH
# ======================================================
today = pd.Timestamp.today()
future_meetings = fomc_dates[fomc_dates >= today]

if future_meetings.empty:
    st.error("No future FOMC meetings found.")
    st.stop()

first_meeting = future_meetings.min()

# Anchor = month BEFORE first meeting month
anchor_month = (
    first_meeting.to_period("M").to_timestamp()
    - pd.DateOffset(months=1)
)

# Build 13 months: anchor + next 12
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
    end = m + pd.offsets.MonthEnd(1)
    days = (end - m).days + 1

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

prev = anchor_rate
premia = []

for _, r in work_df.iterrows():
    if r["Post"] == 0:
        premia.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new_rate = (r["Month Rate"] - w_pre * prev) / w_post
    premia.append((new_rate - prev) * 100)
    prev = new_rate

work_df["Meeting Premium (bps)"] = premia
work_df["Policy Path"] = anchor_rate + np.cumsum(work_df["Meeting Premium (bps)"]) / 100

# ======================================================
# 6. DISPLAY MEETING PREMIA
# ======================================================
st.subheader("Implied FOMC Meeting Premia (from ZQ)")

st.dataframe(
    work_df[
        ["Month", "Meeting Date", "Meeting Premium (bps)", "Policy Path"]
    ],
    use_container_width=True
)

# ======================================================
# 7. SR3 QUARTERS (EDIT ONCE PER YEAR)
# ======================================================
sr3_quarters = pd.DataFrame([
    ("SR3H6", "2026-03-18", "2026-06-17"),
    ("SR3M6", "2026-06-17", "2026-09-16"),
    ("SR3U6", "2026-09-16", "2026-12-16"),
    ("SR3Z6", "2026-12-16", "2027-03-17"),
], columns=["SR3", "Start", "End"])

sr3_quarters["Start"] = pd.to_datetime(sr3_quarters["Start"])
sr3_quarters["End"] = pd.to_datetime(sr3_quarters["End"])

# ======================================================
# 8. MAP MEETING PREMIA → SR3 (FULL TRANSPARENCY)
# ======================================================
sr3_detail_rows = []
sr3_summary_rows = []

for _, q in sr3_quarters.iterrows():
    total_days = (q["End"] - q["Start"]).days
    total_contribution_bps = 0.0

    for _, m in work_df.iterrows():
        meet = m["Meeting Date"]
        if pd.isna(meet):
            continue
        if meet < q["Start"]:
            continue
        if meet >= q["End"]:
            break

        days_after = (q["End"] - meet).days
        weight = days_after / total_days
        contrib_bps = weight * m["Meeting Premium (bps)"]

        total_contribution_bps += contrib_bps

        sr3_detail_rows.append({
            "SR3": q["SR3"],
            "Meeting Date": meet.date(),
            "Meeting Premium (bps)": round(m["Meeting Premium (bps)"], 2),
            "Weight in SR3": round(weight, 4),
            "Contribution (bps)": round(contrib_bps, 2),
        })

    implied_rate = anchor_rate + total_contribution_bps / 100
    implied_price = 100 - implied_rate

    sr3_summary_rows.append({
        "SR3": q["SR3"],
        "Anchor Rate (%)": round(anchor_rate, 3),
        "Total Meeting Impact (bps)": round(total_contribution_bps, 2),
        "Implied SR3 Rate (%)": round(implied_rate, 3),
        "Implied SR3 Price": round(implied_price, 3),
    })

sr3_detail_df = pd.DataFrame(sr3_detail_rows)
sr3_summary_df = pd.DataFrame(sr3_summary_rows)

# ======================================================
# 9. DISPLAY SR3 OUTPUT
# ======================================================
st.subheader("SR3 – Meeting Premium Decomposition")
st.dataframe(sr3_detail_df, use_container_width=True)

st.subheader("SR3 – Implied Rates & Prices")
st.dataframe(sr3_summary_df, use_container_width=True)

# ======================================================
# 10. FOOTER
# ======================================================
st.caption(
    f"As of {date.today()} | "
    f"Anchor month: {anchor_label} | "
    f"Anchor rate: {anchor_rate:.2f}%"
)
