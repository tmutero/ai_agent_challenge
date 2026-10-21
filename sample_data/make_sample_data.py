"""
Generate the bundled sample documents for a fictional asset manager,
"Kestrel Asset Management". The RAG code knows nothing about them: they are
loaded exactly like any user upload.

    funds.csv                fund factsheet table (CSV)
    fee_schedule.json        platform fee schedule (nested JSON)
    fica_requirements.csv    client onboarding / FICA document matrix (CSV)
    investment_policy.pdf    suitability + asset-limit policy (tabular PDF + prose)

All data is synthetic. Values are deliberately distinct (no two funds share a TER,
etc.) so grounding can be checked: a wrong number in an answer is detectable.

    python sample_data/make_sample_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).parent

FUND_COLUMNS = [
    "fund_code", "fund_name", "asset_class", "risk_rating", "reg28_compliant",
    "ter_pct", "min_lump_sum_zar", "min_monthly_zar", "inception_date",
    "return_1y_pct", "return_3y_pct", "return_5y_pct", "benchmark",
    "portfolio_manager", "distribution_frequency",
]

# Empty return cells are intentional: funds younger than 3/5 years have no
# track record, and the assistant must say so rather than invent a number.
FUNDS = [
    ["KST-MM01", "Kestrel Money Market Fund", "Money Market", "Low", "Yes", 0.29, 1000, 250, "2009-03-02", 8.42, 7.31, 6.55, "STeFI Call Deposit Index", "Naledi Khumalo", "Monthly"],
    ["KST-INC02", "Kestrel Income Fund", "Multi-Asset Income", "Low-Medium", "Yes", 0.62, 5000, 500, "2011-07-01", 10.18, 8.94, 8.12, "STeFI Composite + 1%", "Naledi Khumalo", "Quarterly"],
    ["KST-BND03", "Kestrel Bond Fund", "SA Bonds", "Medium", "No", 0.55, 10000, 500, "2012-01-16", 13.64, 9.87, 8.71, "FTSE/JSE All Bond Index", "Pieter van Wyk", "Semi-annual"],
    ["KST-CON04", "Kestrel Conservative Balanced Fund", "Multi-Asset Low Equity", "Low-Medium", "Yes", 1.08, 10000, 500, "2010-05-03", 11.23, 9.05, 8.36, "CPI + 3%", "Thandiwe Mokoena", "Semi-annual"],
    ["KST-BAL05", "Kestrel Balanced Fund", "Multi-Asset High Equity", "Medium", "Yes", 1.21, 10000, 500, "2008-09-01", 14.87, 10.42, 9.18, "ASISA SA MA High Equity Category Average", "Thandiwe Mokoena", "Semi-annual"],
    ["KST-EQ06", "Kestrel SA Equity Fund", "SA Equity", "High", "No", 1.37, 20000, 1000, "2007-02-12", 17.95, 11.86, 10.04, "FTSE/JSE Capped SWIX", "Sipho Dlamini", "Annual"],
    ["KST-DIV07", "Kestrel Dividend Growth Fund", "SA Equity", "Medium-High", "No", 1.14, 20000, 1000, "2014-10-01", 15.32, 12.47, 9.63, "FTSE/JSE Dividend Plus Index", "Sipho Dlamini", "Semi-annual"],
    ["KST-PRP08", "Kestrel Property Fund", "SA Real Estate", "High", "No", 0.97, 10000, 500, "2013-04-02", 21.06, 8.33, 3.92, "FTSE/JSE SA Listed Property Index", "Ayesha Patel", "Quarterly"],
    ["KST-GEQ09", "Kestrel Global Equity Feeder Fund", "Global Equity", "High", "No", 1.49, 20000, 1000, "2015-06-01", 19.74, 13.58, 12.21, "MSCI All Country World Index (ZAR)", "James Whitfield", "Annual"],
    ["KST-GBL10", "Kestrel Global Balanced Feeder Fund", "Global Multi-Asset", "Medium-High", "No", 1.33, 10000, 500, "2016-03-01", 12.96, 9.71, 8.45, "60% MSCI ACWI / 40% Bloomberg Global Aggregate (ZAR)", "James Whitfield", "Annual"],
    ["KST-INF11", "Kestrel Infrastructure Income Fund", "Multi-Asset Income", "Medium", "Yes", 0.88, 50000, 2000, "2021-08-02", 11.87, 9.94, "", "CPI + 4%", "Ayesha Patel", "Quarterly"],
    ["KST-ESG12", "Kestrel Sustainable Equity Fund", "SA Equity", "High", "No", 1.42, 20000, 1000, "2023-02-01", 16.48, "", "", "FTSE/JSE Responsible Investment Index", "Lindiwe Ndlovu", "Annual"],
    ["KST-TFS13", "Kestrel Tax-Free Balanced Fund", "Multi-Asset High Equity", "Medium", "Yes", 0.74, 500, 250, "2017-03-01", 14.12, 10.09, 8.97, "CPI + 5%", "Thandiwe Mokoena", "None (reinvested)"],
    ["KST-ABS14", "Kestrel Absolute Return Fund", "Multi-Asset Flexible", "Medium", "No", 1.56, 50000, 2000, "2019-11-01", 12.38, 10.77, 9.29, "CPI + 4% over rolling 3 years", "Pieter van Wyk", "Semi-annual"],
]

FEE_SCHEDULE = {
    "document": "Kestrel Platform Fee Schedule",
    "effective_date": "2026-03-01",
    "currency": "ZAR",
    "vat_note": "All fees exclude VAT at 15% unless stated otherwise.",
    "products": [
        {
            "product_code": "UT",
            "product": "Unit Trust Investment Account",
            "admin_fee_tiers": [
                {"from_zar": 0, "to_zar": 250000, "annual_fee_pct": 0.50},
                {"from_zar": 250000, "to_zar": 1000000, "annual_fee_pct": 0.35},
                {"from_zar": 1000000, "to_zar": None, "annual_fee_pct": 0.20},
            ],
            "switch_fee_zar": 0,
            "max_free_switches_per_year": 12,
            "withdrawal_notice_days": 2,
            "max_advice_fee_pct": 1.00,
        },
        {
            "product_code": "RA",
            "product": "Retirement Annuity",
            "admin_fee_tiers": [
                {"from_zar": 0, "to_zar": 500000, "annual_fee_pct": 0.45},
                {"from_zar": 500000, "to_zar": 2000000, "annual_fee_pct": 0.30},
                {"from_zar": 2000000, "to_zar": None, "annual_fee_pct": 0.18},
            ],
            "switch_fee_zar": 0,
            "max_free_switches_per_year": 4,
            "minimum_retirement_age": 55,
            "max_advice_fee_pct": 0.75,
            "reg28_applies": True,
        },
        {
            "product_code": "TFSA",
            "product": "Tax-Free Savings Account",
            "admin_fee_tiers": [{"from_zar": 0, "to_zar": None, "annual_fee_pct": 0.25}],
            "annual_contribution_limit_zar": 46000,
            "lifetime_contribution_limit_zar": 500000,
            "over_contribution_penalty_pct": 40,
            "switch_fee_zar": 0,
            "max_free_switches_per_year": 12,
            "max_advice_fee_pct": 0.50,
        },
        {
            "product_code": "LA",
            "product": "Living Annuity",
            "admin_fee_tiers": [
                {"from_zar": 0, "to_zar": 1000000, "annual_fee_pct": 0.40},
                {"from_zar": 1000000, "to_zar": 3000000, "annual_fee_pct": 0.27},
                {"from_zar": 3000000, "to_zar": None, "annual_fee_pct": 0.15},
            ],
            "drawdown_range_pct": {"min": 2.5, "max": 17.5},
            "drawdown_change_frequency": "Once per year on the policy anniversary",
            "switch_fee_zar": 150,
            "max_free_switches_per_year": 2,
            "max_advice_fee_pct": 0.85,
        },
    ],
}

FICA_COLUMNS = [
    "client_type", "required_documents", "proof_of_address_max_age_months",
    "verification_method", "enhanced_due_diligence", "review_cycle_months",
    "onboarding_sla_business_days",
]
FICA = [
    ["Individual - SA resident", "Green barcoded ID or smart ID card; proof of address; bank confirmation letter", 3, "Home Affairs online ID verification", "No", 36, 2],
    ["Individual - foreign national", "Valid passport; valid SA visa or permit; proof of address; bank confirmation letter", 3, "Passport scan + manual compliance review", "Yes", 12, 5],
    ["Minor", "Unabridged birth certificate; guardian's ID; guardian's proof of address", 3, "Home Affairs verification of guardian", "No", 36, 3],
    ["Trust", "Letters of authority; trust deed; IDs of all trustees and beneficiaries; resolution authorising investment", 3, "Master of the High Court record check", "Yes", 12, 7],
    ["Company (Pty) Ltd", "CIPC registration certificate; IDs of directors and shareholders above 25%; company resolution; proof of business address", 3, "CIPC registry check + beneficial ownership review", "Yes", 12, 7],
    ["Politically exposed person (PEP)", "As per individual type plus source-of-funds declaration and senior management approval", 3, "Manual compliance review + PEP database screening", "Yes", 6, 10],
]

SUITABILITY_HEADER = ["Client risk profile", "Max fund risk rating", "Max equity exposure", "Max offshore exposure", "Min horizon (years)", "Review frequency"]
SUITABILITY = [
    ["Conservative", "Low-Medium", "25%", "10%", "1", "Annual"],
    ["Moderately Conservative", "Medium", "40%", "20%", "3", "Annual"],
    ["Moderate", "Medium", "60%", "30%", "5", "Annual"],
    ["Moderately Aggressive", "Medium-High", "75%", "40%", "5", "Semi-annual"],
    ["Aggressive", "High", "100%", "45%", "7", "Semi-annual"],
]

LIMITS_HEADER = ["Asset class", "Max % of retirement portfolio", "Single-issuer limit", "Notes"]
LIMITS = [
    ["Equities (total)", "75%", "15% (large cap) / 10% (other)", "Includes offshore equities"],
    ["Listed property", "25%", "5%", "Includes REITs"],
    ["Offshore assets", "45%", "n/a", "Excludes Africa allowance"],
    ["Africa ex-SA", "10%", "n/a", "Additional to offshore limit"],
    ["Private equity", "15%", "2.5% per fund", "Board approval required"],
    ["Hedge funds", "10%", "2.5% per fund", "Fund-of-hedge-funds: 5% per fund"],
    ["Commodities", "10%", "5% (gold) / 2.5% (other)", "Exchange-traded only"],
    ["Infrastructure", "45%", "15% per issuer", "Across debt and equity combined"],
]

POLICY_INTRO = (
    "Kestrel Asset Management - Investor Suitability and Investment Limits Policy. "
    "Policy owner: Head of Compliance. Version 4.2, approved by the Investment Committee on 2026-02-18. "
    "Purpose: this policy sets the maximum risk an adviser may recommend for each client risk profile "
    "and the asset-class limits that apply to all retirement products (Retirement Annuity and preservation funds). "
    "A client's risk profile must be re-assessed at least at the review frequency below, and immediately after a "
    "material life event (retirement, divorce, death of a spouse, or a change of more than 30% in investable assets)."
)
POLICY_BREACH = (
    "Breaches. A passive breach of an asset-class limit (caused by market movement) must be corrected within "
    "12 months. An active breach (caused by a switch or new contribution) must be reported to Compliance within "
    "2 business days and corrected within 30 days. Advisers may not recommend a fund whose risk rating exceeds the "
    "client's maximum fund risk rating without a signed client deviation letter."
)


def write_funds() -> None:
    with open(DATA / "funds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FUND_COLUMNS)
        w.writerows(FUNDS)


def write_fees() -> None:
    (DATA / "fee_schedule.json").write_text(json.dumps(FEE_SCHEDULE, indent=2), encoding="utf-8")


def write_fica() -> None:
    with open(DATA / "fica_requirements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FICA_COLUMNS)
        w.writerows(FICA)


def write_policy_pdf() -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(DATA / "investment_policy.pdf"), pagesize=landscape(A4),
                            title="Investor Suitability and Investment Limits Policy")
    style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ])
    story = [
        Paragraph("Investor Suitability and Investment Limits Policy", styles["Title"]),
        Paragraph(POLICY_INTRO, styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Table 1 - Risk profile suitability matrix", styles["Heading2"]),
        Table([SUITABILITY_HEADER] + SUITABILITY, style=style),
        Spacer(1, 12),
        Paragraph("Table 2 - Retirement product asset-class limits", styles["Heading2"]),
        Table([LIMITS_HEADER] + LIMITS, style=style),
        Spacer(1, 12),
        Paragraph(POLICY_BREACH, styles["BodyText"]),
    ]
    doc.build(story)


def main() -> None:
    write_funds()
    write_fees()
    write_fica()
    write_policy_pdf()
    print(f"Wrote knowledge base to {DATA}")


if __name__ == "__main__":
    main()
