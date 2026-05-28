"""Scan universe: ~190 tickers across TSX 60, TSX mid-caps, CIBC CDRs, and ETFs.

Composition (see DECISIONS.md for rationale):
  - TSX 60 constituents:     ~60 names (.TO)
  - Additional TSX mid-caps: ~46 names (.TO)
  - CIBC CDR catalog:        ~65 names (.NE) — verified resolvable, reused from cdr_mapping.py
  - TSX-listed ETFs:         ~20 names (.TO)
"""

# ── TSX 60 constituents ──────────────────────────────────────────────────────
TSX60_TICKERS: list[str] = [
    # Banks
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    # Insurance / diversified financials
    "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "FFH.TO", "POW.TO",
    # Energy producers
    "CNQ.TO", "SU.TO", "CVE.TO", "IMO.TO",
    # Pipelines / midstream
    "ENB.TO", "TRP.TO", "PPL.TO",
    # Railroads
    "CNR.TO", "CP.TO",
    # Telecom
    "BCE.TO", "T.TO", "RCI-B.TO",
    # Technology
    "SHOP.TO", "CSU.TO", "OTEX.TO",
    # Industrials
    "WSP.TO", "CAE.TO", "TIH.TO", "TFII.TO", "GFL.TO", "WCN.TO",
    # Materials / mining
    "ABX.TO", "AEM.TO", "FNV.TO", "WPM.TO", "K.TO", "FM.TO", "NTR.TO",
    # Consumer
    "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "QSR.TO", "SAP.TO", "CCL-B.TO",
    # Utilities
    "FTS.TO", "EMA.TO", "H.TO", "NPI.TO",
    # Diversified / alt managers
    "BAM.TO", "BN.TO", "BIP-UN.TO", "BEP-UN.TO",
    # Media / info services
    "TRI.TO", "X.TO", "GIB-A.TO",
    # Automotive
    "MG.TO",
    # Other
    "LIF.TO",
]

# ── Additional liquid TSX mid-caps ───────────────────────────────────────────
TSX_MIDCAP_TICKERS: list[str] = [
    # Consumer / retail
    "AC.TO", "CTC-A.TO", "EMP-A.TO", "PBH.TO", "MTY.TO", "TOY.TO",
    # Technology
    "BB.TO", "LSPD.TO", "KXS.TO", "DCBO.TO", "CLS.TO", "DSG.TO",
    "MDA.TO",
    # Financials
    "IAG.TO", "LB.TO", "ONEX.TO", "ECN.TO", "EFN.TO",
    # Industrials / engineering
    "STN.TO", "CIGI.TO",
    # Energy (not in TSX 60)
    "OVV.TO", "BTE.TO", "ARX.TO", "WCP.TO",
    "PEY.TO", "TVE.TO", "BIR.TO", "TOU.TO", "VET.TO",
    # Pipelines / utilities
    "KEY.TO", "ALA.TO", "SPB.TO",
    # Mining
    "AGI.TO", "OR.TO", "LUN.TO", "TECK-B.TO",
    # REITs
    "DIR-UN.TO", "REI-UN.TO", "AP-UN.TO", "SRU-UN.TO",
    "HR-UN.TO", "CAR-UN.TO", "GRT-UN.TO",
    # Other
    "BLDP.TO", "FRU.TO", "PSK.TO",
]

# ── TSX-listed ETFs ──────────────────────────────────────────────────────────
TSX_ETF_TICKERS: list[str] = [
    "XIC.TO",    # iShares Core S&P/TSX Capped Composite
    "XIU.TO",    # iShares S&P/TSX 60
    "XEG.TO",    # iShares S&P/TSX Capped Energy
    "XFN.TO",    # iShares S&P/TSX Capped Financials
    "ZEB.TO",    # BMO Equal Weight Banks
    "XIT.TO",    # iShares S&P/TSX Capped Info Tech
    "XMA.TO",    # iShares S&P/TSX Capped Materials
    "XEQT.TO",  # iShares Core Equity ETF
    "VEQT.TO",  # Vanguard All-Equity ETF
    "VFV.TO",   # Vanguard S&P 500 Index (CAD)
    "ZSP.TO",   # BMO S&P 500 Index
    "XSP.TO",   # iShares Core S&P 500 (CAD-Hedged)
    "QQC-F.TO", # Invesco Nasdaq 100 (CAD-Hedged)
    "ZQQ.TO",   # BMO Nasdaq 100
    "ZCN.TO",   # BMO S&P/TSX Capped Composite
    "ZDV.TO",   # BMO Canadian Dividend
    "XDV.TO",   # iShares Canadian Select Dividend
    "ZWB.TO",   # BMO Covered Call Canadian Banks
    "XGRO.TO",  # iShares Core Growth ETF
    "VCN.TO",   # Vanguard FTSE Canada All Cap
]


def get_scan_universe() -> list[str]:
    """Return the full deduplicated scan universe (~220 tickers).

    Order: TSX 60, TSX mid-caps, CIBC CDRs, ETFs.
    Duplicates are removed (first occurrence wins).
    """
    from sterling.cdr_mapping import CDR_TICKERS

    all_tickers = TSX60_TICKERS + TSX_MIDCAP_TICKERS + list(CDR_TICKERS) + TSX_ETF_TICKERS
    seen: set[str] = set()
    result: list[str] = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result
