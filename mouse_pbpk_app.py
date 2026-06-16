from __future__ import annotations

import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mouse PBPK Simulator", layout="wide")

CUSTOM_CSS = """
<style>
.header {
  background: linear-gradient(90deg, rgba(11,95,255,0.10), rgba(11,95,255,0.02), rgba(11,95,255,0.10));
  background-size: 200% 200%;
  animation: gradient 10s ease infinite;
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
}
@keyframes gradient {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(11,95,255,0.25);
  color: rgba(11,95,255,1);
  background: rgba(11,95,255,0.06);
  font-size: 12px;
  margin-left: 10px;
}
.smallnote {
  color: rgba(17,24,39,0.68);
  font-size: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.container():
    c1, c2 = st.columns([0.18, 0.82], vertical_alignment="center")
    with c1:
        st.empty()
    with c2:
        st.markdown(
            '<div class="header">'
            '<div style="display:flex;align-items:center;gap:10px;">'
            '<h2 style="margin:0;">Mouse PBPK Simulator</h2>'
            '<span class="badge">Generic compound workflow</span>'
            '</div>'
            '<div class="smallnote">Mouse PBPK model with Rowland-Rodgers Kp estimation, microsomal half-life based hepatic clearance, and observed-data overlay.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

st.caption("Research / education use only. Not validated for clinical decision-making.")


# -----------------------------------------------------------------------------
# Model assumptions and tissue composition constants
# -----------------------------------------------------------------------------
# Physiology is based on the reference mouse code. Volumes are scaled from a
# 25 g mouse baseline. Blood flows are scaled allometrically by BW^0.75.

BASE_BW_G = 25.0
BASE_VOLUMES_ML = {
    "adipose": 6.8,
    "bone": 2.68,
    "brain": 0.4125,
    "heart": 0.125,
    "kidney": 0.4175,
    "liver": 1.3725,
    "lung": 0.1825,
    "muscle": 9.6,
    "skin": 4.1325,
    "spleen": 0.0875,
    "gut": 1.055,
    "rest": 0.135,
    "arterial_blood": 0.57,
    "venous_blood": 1.13,
}

BASE_FLOWS_ML_H = {
    "adipose": 0.0013 * 60.0 * 60.0,
    "bone": None,
    "brain": 0.0076 * 60.0 * 60.0,
    "heart": 0.0047 * 60.0 * 60.0,
    "kidney": 0.0213 * 60.0 * 60.0,
    "liver": 0.0335 * 60.0 * 60.0,
    "lung": 0.1535 * 60.0 * 60.0,
    "muscle": 0.0152 * 60.0 * 60.0,
    "skin": None,
    "spleen": 0.0015 * 60.0 * 60.0,
    "gut": 0.025 * 60.0 * 60.0,
    "rest": 0.07 * 60.0 * 60.0,
}

# Tissue composition fractions from the provided reference code.
TISSUE_COMPOSITION = {
    "adipose": dict(Few=0.135, Fiw=0.017, Fnl=0.0016, Fnp=0.853, albumin_tissue=0.049),
    "bone": dict(Few=0.100, Fiw=0.346, Fnl=0.0017, Fnp=0.017, albumin_tissue=0.157),
    "brain": dict(Few=0.162, Fiw=0.620, Fnl=0.0015, Fnp=0.039, albumin_tissue=0.048),
    "gut": dict(Few=0.282, Fiw=0.475, Fnl=0.0125, Fnp=0.038, albumin_tissue=0.158),
    "heart": dict(Few=0.320, Fiw=0.475, Fnl=0.0111, Fnp=0.014, albumin_tissue=0.157),
    "kidney": dict(Few=0.273, Fiw=0.483, Fnl=0.0242, Fnp=0.012, albumin_tissue=0.130),
    "liver": dict(Few=0.161, Fiw=0.573, Fnl=0.0240, Fnp=0.014, albumin_tissue=0.086),
    "lung": dict(Few=0.336, Fiw=0.446, Fnl=0.0128, Fnp=0.022, albumin_tissue=0.212),
    "muscle": dict(Few=0.118, Fiw=0.630, Fnl=0.0072, Fnp=0.010, albumin_tissue=0.064),
    "skin": dict(Few=0.382, Fiw=0.291, Fnl=0.0044, Fnp=0.060, albumin_tissue=0.277),
    "spleen": dict(Few=0.207, Fiw=0.579, Fnl=0.0113, Fnp=0.0077, albumin_tissue=0.097),
    "rest": dict(Few=0.135, Fiw=0.640, Fnl=0.0092, Fnp=0.030, albumin_tissue=0.122),
}

# Fixed physiological constants used by the RR equation and hepatic scaling.
PHI_W = 7.0
PH_P = 7.4
ALBUMIN_PLASMA = 0.05
ALBUMIN_TISSUE_DEFAULT = 0.10
MPPGL_MOUSE = 45.0  # mg microsomal protein / g liver
MICROSOME_PROTEIN_CONC = 0.5  # mg/mL in the microsomal incubation

# The model states are amount compartments (mg).
STATE = [
    "Acentral",  # venous / plasma reference compartment
    "Aadipose",
    "Abone",
    "Abrain",
    "Aheart",
    "Akidney",
    "Aliver",
    "Alung",
    "Amuscle",
    "Askin",
    "Aspleen",
    "Agut",
    "Arest",
    "Agut_lumen",
]


@dataclass
class MouseParams:
    bw_g: float = 25.0
    acidic_pkas: str = ""
    basic_pkas: str = ""
    logp: float = 2.5
    fup: float = 0.1
    rbp: float = 1.5
    liver_microsome_t_half_min: float = 45.0
    p_hi_w: float = PHI_W
    p_hp: float = PH_P
    albumin_plasma: float = ALBUMIN_PLASMA
    albumin_tissue: float = ALBUMIN_TISSUE_DEFAULT
    microsome_protein_conc_mg_per_ml: float = MICROSOME_PROTEIN_CONC
    mppgl: float = MPPGL_MOUSE
    iv_dose_mg: float = 0.0
    oral_dose_mg: float = 10.0
    route: str = "Oral"
    dose_interval_h: float = 24.0
    n_doses: int = 1
    simulation_days: float = 7.0


PROPRANOLOL_EXAMPLE = {
    "bw_g": 25.0,
    "acidic_pkas": "",
    "basic_pkas": "9.5",
    "logp": 3.03,
    "fup": 0.10,
    "rbp": 1.5,
    "liver_microsome_t_half_min": 45.0,
    "route": "Oral",
    "oral_dose_mg": 0.25,
    "iv_dose_mg": 0.25,
    "dose_interval_h": 24.0,
    "n_doses": 1,
    "simulation_days": 7.0,
}


def build_propranolol_reference_band(simulation_days: float) -> pd.DataFrame:
    """Starter default overlay based on mouse propranolol literature ranges.

    The accessible mouse literature used here reports propranolol plasma levels
    in the ~10-140 ng/mL range under chronic dosing conditions, rather than a
    digitized full time-course table. This band is meant as a demo overlay that
    can be replaced by an uploaded CSV.
    """
    t = np.linspace(0.0, float(simulation_days) * 24.0, 200)
    return pd.DataFrame({
        "time_h": t,
        "lower_ng_mL": np.full_like(t, 10.0, dtype=float),
        "upper_ng_mL": np.full_like(t, 140.0, dtype=float),
        "reference": np.full_like(t, "mouse literature range", dtype=object),
    })


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def log10_safe(x: float) -> float:
    return float(np.log10(max(x, 1e-300)))


def parse_pka_list(text_value: str) -> List[float]:
    """Parse a comma/space/semicolon separated pKa string into floats."""
    if text_value is None:
        return []
    parts = [p.strip() for p in re.split(r"[;,\s]+", str(text_value).strip()) if p.strip()]
    out: List[float] = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def infer_ionization_type(acidic_pkas: Sequence[float], basic_pkas: Sequence[float]) -> str:
    has_acid = len(acidic_pkas) > 0
    has_base = len(basic_pkas) > 0
    if has_acid and has_base:
        return "zwitterion"
    if has_acid:
        return "acid"
    if has_base:
        return "base"
    return "neutral"


def unionized_fraction_acid(pH: float, pkas: Sequence[float]) -> float:
    frac = 1.0
    for pka in pkas:
        frac *= 1.0 / (1.0 + 10.0 ** (pH - float(pka)))
    return float(frac)


def unionized_fraction_base(pH: float, pkas: Sequence[float]) -> float:
    frac = 1.0
    for pka in pkas:
        frac *= 1.0 / (1.0 + 10.0 ** (float(pka) - pH))
    return float(frac)


def to_representation_t_half_min(category: str) -> float:
    mapping = {
        "Low (1-30 min)": 15.0,
        "Medium (30-60 min)": 45.0,
        "High (60-120 min)": 90.0,
    }
    return mapping.get(category, 45.0)


def compute_intracell_ionization(pka: float, p_hi_w: float) -> float:
    return abs(p_hi_w - pka)


def compute_plasma_ionization(pka: float, p_hp: float) -> float:
    return abs(p_hp - pka)


def compute_rr_kp(
    tissue: str,
    ionization_type: str,
    acidic_pkas: Sequence[float],
    basic_pkas: Sequence[float],
    logp: float,
    fup: float,
    p_hi_w: float = PHI_W,
    p_hp: float = PH_P,
    albumin_tissue: float = ALBUMIN_TISSUE_DEFAULT,
    albumin_plasma: float = ALBUMIN_PLASMA,
) -> float:
    """Rowland-Rodgers style unbound tissue partition coefficient and Kp.

    The generic composition terms are taken from the provided mouse reference.
    The equation branch is selected from the inferred ionization class.
    This implementation supports neutral, acid, base, and zwitterion inputs.
    """
    comp = TISSUE_COMPOSITION[tissue]
    pc = 10.0 ** float(logp)

    ionization_type = str(ionization_type).lower().strip()
    if ionization_type == "neutral":
        plasma_union = 1.0
        intracell_union = 1.0
    elif ionization_type == "acid":
        plasma_union = max(unionized_fraction_acid(p_hp, acidic_pkas), 1e-12)
        intracell_union = max(unionized_fraction_acid(p_hi_w, acidic_pkas), 1e-12)
    elif ionization_type == "base":
        plasma_union = max(unionized_fraction_base(p_hp, basic_pkas), 1e-12)
        intracell_union = max(unionized_fraction_base(p_hi_w, basic_pkas), 1e-12)
    else:  # zwitterion or mixed-ionization compound
        plasma_union = max(
            unionized_fraction_acid(p_hp, acidic_pkas) * unionized_fraction_base(p_hp, basic_pkas),
            1e-12,
        )
        intracell_union = max(
            unionized_fraction_acid(p_hi_w, acidic_pkas) * unionized_fraction_base(p_hi_w, basic_pkas),
            1e-12,
        )

    # Generic Rodgers/Rowland-style structure adapted to the compound class.
    term_water = (intracell_union / plasma_union) * comp["Fiw"]
    term_lipid = (pc * comp["Fnl"] + (0.3 * pc + 0.7) * comp["Fnp"]) / plasma_union
    term_albumin = ((1.0 / max(fup, 1e-12)) - 1.0 - term_lipid) * (albumin_tissue / albumin_plasma)
    kpu = comp["Few"] + term_water + term_lipid + term_albumin
    kp = max(1e-12, kpu * fup)
    return float(kp)


def compute_kps(params: MouseParams) -> Dict[str, float]:
    acidic_pkas = parse_pka_list(params.acidic_pkas)
    basic_pkas = parse_pka_list(params.basic_pkas)
    ionization_type = infer_ionization_type(acidic_pkas, basic_pkas)
    kps = {}
    for tissue in TISSUE_COMPOSITION:
        kps[tissue] = compute_rr_kp(
            tissue=tissue,
            ionization_type=ionization_type,
            acidic_pkas=acidic_pkas,
            basic_pkas=basic_pkas,
            logp=params.logp,
            fup=params.fup,
            p_hi_w=params.p_hi_w,
            p_hp=params.p_hp,
            albumin_tissue=params.albumin_tissue,
            albumin_plasma=params.albumin_plasma,
        )
    return kps


def scale_volume(base_ml: float, bw_g: float) -> float:
    return float(base_ml * (bw_g / BASE_BW_G))


def scale_flow(base_ml_h: float, bw_g: float) -> float:
    # Slightly sublinear scaling for flows.
    return float(base_ml_h * (bw_g / BASE_BW_G) ** 0.75)


def build_mouse_physiology(bw_g: float) -> Dict[str, float]:
    vols = {f"V_{k}": scale_volume(v, bw_g) for k, v in BASE_VOLUMES_ML.items()}

    # Fill missing blood flow placeholders.
    qcard = 0.275 * (bw_g ** 0.75) * 60.0  # mL/h
    q = {f"Q_{k}": scale_flow(v, bw_g) for k, v in BASE_FLOWS_ML_H.items() if v is not None}

    # Provide a few flows by fraction of cardiac output to keep the model balanced.
    q["Q_bone"] = 0.11 * qcard
    q["Q_skin"] = 18.0 * 60.0 * bw_g / 100.0
    q["Q_heart"] = q["Q_heart"] if "Q_heart" in q else 0.0047 * 60.0 * 60.0 * (bw_g / BASE_BW_G) ** 0.75

    # Recompute a few missing values with the same scaling.
    if "Q_adipose" not in q:
        q["Q_adipose"] = 0.0013 * 60.0 * 60.0 * (bw_g / BASE_BW_G) ** 0.75

    # blood volume (central)
    q["Qcard"] = qcard
    q["Vcentral"] = vols["V_arterial_blood"] + vols["V_venous_blood"]
    q["VLiverWeight_g"] = 1.2 * (bw_g / 25.0)
    q["QH"] = 131.0 * 60.0 * bw_g / 100.0  # mL/h, reference liver blood flow proxy
    return {**vols, **q}


def hepatic_clearance_from_t_half(params: MouseParams, phys: Dict[str, float]) -> Dict[str, float]:
    """Convert microsomal t1/2 to mouse hepatic clearance using the reference equations."""
    t_half_min = max(1e-6, float(params.liver_microsome_t_half_min))
    t_half_hr = t_half_min / 60.0
    clint_invitro_per_hr = (math.log(2.0) / t_half_hr) / params.microsome_protein_conc_mg_per_ml
    clint_h = clint_invitro_per_hr * params.mppgl * phys["VLiverWeight_g"]
    qh = phys["QH"]
    clh = (qh * params.fup * clint_h) / (qh + (params.fup * clint_h))
    return {
        "CLint_invitro_per_hr": clint_invitro_per_hr,
        "CLintH_mL_h": clint_h,
        "CLH_mL_h": clh,
        "t_half_hr": t_half_hr,
    }


def compound_ionization_summary(params: MouseParams) -> Dict[str, object]:
    acidic_pkas = parse_pka_list(params.acidic_pkas)
    basic_pkas = parse_pka_list(params.basic_pkas)
    ionization_type = infer_ionization_type(acidic_pkas, basic_pkas)
    return {
        "ionization_type": ionization_type,
        "acidic_pkas": acidic_pkas,
        "basic_pkas": basic_pkas,
    }


def build_dose_events(route: str, oral_dose_mg: float, iv_dose_mg: float, dose_interval_h: float, n_doses: int) -> List[Tuple[float, float]]:
    events: List[Tuple[float, float]] = []
    for i in range(max(1, int(n_doses))):
        t = i * float(dose_interval_h)
        dose = float(iv_dose_mg if route == "IV" else oral_dose_mg)
        events.append((t, dose))
    return events


def add_dose(y: np.ndarray, route: str, dose_mg: float) -> np.ndarray:
    y2 = np.array(y, dtype=float, copy=True)
    if route == "IV":
        y2[STATE.index("Acentral")] += float(dose_mg)
    else:
        y2[STATE.index("Agut_lumen")] += float(dose_mg)
    return y2


# -----------------------------------------------------------------------------
# ODE model
# -----------------------------------------------------------------------------
def rhs(t: float, y: np.ndarray, params: MouseParams, phys: Dict[str, float], kps: Dict[str, float], clh: float) -> np.ndarray:
    Acentral, Aadipose, Abone, Abrain, Aheart, Akidney, Aliver, Alung, Amuscle, Askin, Aspleen, Agut, Arest, Agut_lumen = y

    # Volumes (mL)
    Vc = phys["Vcentral"]
    V_ad = phys["V_adipose"]
    V_bo = phys["V_bone"]
    V_br = phys["V_brain"]
    V_he = phys["V_heart"]
    V_ki = phys["V_kidney"]
    V_li = phys["V_liver"]
    V_lu = phys["V_lung"]
    V_mu = phys["V_muscle"]
    V_sk = phys["V_skin"]
    V_sp = phys["V_spleen"]
    V_gu = phys["V_gut"]
    V_re = phys["V_rest"]

    # Flows (mL/h)
    Q_ad = phys["Q_adipose"]
    Q_bo = phys["Q_bone"]
    Q_br = phys["Q_brain"]
    Q_he = phys["Q_heart"]
    Q_ki = phys["Q_kidney"]
    Q_li = phys["Q_liver"]
    Q_lu = phys["Q_lung"]
    Q_mu = phys["Q_muscle"]
    Q_sk = phys["Q_skin"]
    Q_sp = phys["Q_spleen"]
    Q_gu = phys["Q_gut"]
    Q_re = phys["Q_rest"]

    # Concentrations in mg/mL blood or mg/mL tissue.
    Cp = Acentral / max(Vc, 1e-12)
    Ca = Cp / max(params.rbp, 1e-12)

    C_ad = Aadipose / max(V_ad, 1e-12)
    C_bo = Abone / max(V_bo, 1e-12)
    C_br = Abrain / max(V_br, 1e-12)
    C_he = Aheart / max(V_he, 1e-12)
    C_ki = Akidney / max(V_ki, 1e-12)
    C_li = Aliver / max(V_li, 1e-12)
    C_lu = Alung / max(V_lu, 1e-12)
    C_mu = Amuscle / max(V_mu, 1e-12)
    C_sk = Askin / max(V_sk, 1e-12)
    C_sp = Aspleen / max(V_sp, 1e-12)
    C_gu = Agut / max(V_gu, 1e-12)
    C_re = Arest / max(V_re, 1e-12)
    C_u = Cp * params.fup

    # Perfusion-limited exchange (blood concentration basis).
    f_ad = Q_ad * (Ca - C_ad / max(kps["adipose"], 1e-12) * params.rbp)
    f_bo = Q_bo * (Ca - C_bo / max(kps["bone"], 1e-12) * params.rbp)
    f_br = Q_br * (Ca - C_br / max(kps["brain"], 1e-12) * params.rbp)
    f_he = Q_he * (Ca - C_he / max(kps["heart"], 1e-12) * params.rbp)
    f_ki = Q_ki * (Ca - C_ki / max(kps["kidney"], 1e-12) * params.rbp)
    f_li = Q_li * (Ca - C_li / max(kps["liver"], 1e-12) * params.rbp)
    f_lu = Q_lu * (Ca - C_lu / max(kps["lung"], 1e-12) * params.rbp)
    f_mu = Q_mu * (Ca - C_mu / max(kps["muscle"], 1e-12) * params.rbp)
    f_sk = Q_sk * (Ca - C_sk / max(kps["skin"], 1e-12) * params.rbp)
    f_sp = Q_sp * (Ca - C_sp / max(kps["spleen"], 1e-12) * params.rbp)
    f_gu = Q_gu * (Ca - C_gu / max(kps["gut"], 1e-12) * params.rbp)
    f_re = Q_re * (Ca - C_re / max(kps["rest"], 1e-12) * params.rbp)

    # Hepatic clearance from unbound plasma concentration.
    hep_elim = clh * C_u

    # Simple kidney elimination proxy (fixed physiology, not user input).
    # This keeps the model numerically realistic without exposing another key UI field.
    gfr_ml_h = 0.012 * (params.bw_g ** 0.75) * 60.0
    ren_elim = gfr_ml_h * params.fup * Cp

    d = np.zeros_like(y)

    d[STATE.index("Acentral")] = (
        -f_ad - f_bo - f_br - f_he - f_ki - f_li - f_lu - f_mu - f_sk - f_sp - f_gu - f_re
        - hep_elim - ren_elim
        + params.oral_dose_mg * 0.0
    )

    d[STATE.index("Aadipose")] = f_ad
    d[STATE.index("Abone")] = f_bo
    d[STATE.index("Abrain")] = f_br
    d[STATE.index("Aheart")] = f_he
    d[STATE.index("Akidney")] = f_ki - 0.15 * ren_elim * (C_ki / max(Cp, 1e-12))
    d[STATE.index("Aliver")] = f_li - hep_elim
    d[STATE.index("Alung")] = f_lu
    d[STATE.index("Amuscle")] = f_mu
    d[STATE.index("Askin")] = f_sk
    d[STATE.index("Aspleen")] = f_sp
    d[STATE.index("Agut")] = f_gu
    d[STATE.index("Arest")] = f_re
    d[STATE.index("Agut_lumen")] = -params.oral_dose_mg * 0.0

    # Oral absorption enters central via gut lumen input term in the simulation loop.
    return d


# -----------------------------------------------------------------------------
# Simulation and outputs
# -----------------------------------------------------------------------------
def simulate(params: MouseParams, observed_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    phys = build_mouse_physiology(params.bw_g)
    kps = compute_kps(params)
    hep = hepatic_clearance_from_t_half(params, phys)
    clh = hep["CLH_mL_h"]

    t_end_h = float(params.simulation_days) * 24.0
    dose_events = build_dose_events(params.route, params.oral_dose_mg, params.iv_dose_mg, params.dose_interval_h, params.n_doses)
    dose_times = [t for t, _ in dose_events]
    dose_map = {float(t): float(d) for t, d in dose_events}

    y = np.zeros(len(STATE), dtype=float)
    t_all: List[float] = []
    y_all: List[np.ndarray] = []
    boundaries = sorted(set([0.0, t_end_h] + dose_times))

    # Initial dose at time 0.
    if 0.0 in dose_map:
        y = add_dose(y, params.route, dose_map[0.0])

    for i in range(len(boundaries) - 1):
        t0 = boundaries[i]
        t1 = boundaries[i + 1]
        npts = max(2, int(math.ceil((t1 - t0) / 0.1)) + 1)
        t_eval = np.linspace(t0, t1, npts)

        def local_rhs(tt: float, yy: np.ndarray) -> np.ndarray:
            dy = rhs(tt, yy, params, phys, kps, clh)
            # Oral absorption: gut lumen -> central.
            if params.route == "Oral":
                ka = 0.30
                fab = 1.0
                d_agut = -ka * yy[STATE.index("Agut_lumen")]
                d_ac = ka * yy[STATE.index("Agut_lumen")] * fab
                dy[STATE.index("Agut_lumen")] += d_agut
                dy[STATE.index("Acentral")] += d_ac
            return dy

        sol = solve_ivp(local_rhs, (t0, t1), y, method="LSODA", t_eval=t_eval, rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"Integration failed at interval {t0}-{t1} h: {sol.message}")

        if not t_all:
            t_all.extend(sol.t.tolist())
            y_all.extend(sol.y.T)
        else:
            t_all.extend(sol.t[1:].tolist())
            y_all.extend(sol.y.T[1:])

        y = sol.y[:, -1].copy()
        if t1 in dose_map and t1 != t_end_h:
            y = add_dose(y, params.route, dose_map[t1])

    df = results_dataframe(np.asarray(t_all), np.asarray(y_all), params, phys, kps)
    return df, phys, kps, hep


def results_dataframe(t: np.ndarray, y: np.ndarray, params: MouseParams, phys: Dict[str, float], kps: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(y, columns=STATE)
    df.insert(0, "time_h", t)

    # Concentrations in ng/mL for display.
    for state_name, vol_key, out_col in [
        ("Acentral", "Vcentral", "Plasma (ng/mL)"),
        ("Aadipose", "V_adipose", "Adipose (ng/mL)"),
        ("Abone", "V_bone", "Bone (ng/mL)"),
        ("Abrain", "V_brain", "Brain (ng/mL)"),
        ("Aheart", "V_heart", "Heart (ng/mL)"),
        ("Akidney", "V_kidney", "Kidney (ng/mL)"),
        ("Aliver", "V_liver", "Liver (ng/mL)"),
        ("Alung", "V_lung", "Lung (ng/mL)"),
        ("Amuscle", "V_muscle", "Muscle (ng/mL)"),
        ("Askin", "V_skin", "Skin (ng/mL)"),
        ("Aspleen", "V_spleen", "Spleen (ng/mL)"),
        ("Agut", "V_gut", "Gut (ng/mL)"),
        ("Arest", "V_rest", "Rest of body (ng/mL)"),
    ]:
        df[out_col] = df[state_name] / max(phys[vol_key], 1e-12) * 1e6

    # Useful model outputs.
    for tissue in ["adipose", "bone", "brain", "heart", "kidney", "liver", "lung", "muscle", "skin", "spleen", "gut", "rest"]:
        df[f"Kp_{tissue}"] = kps[tissue]

    df["Model"] = "Mouse PBPK generic RR"
    df["Route"] = params.route
    return df


def auc_trapz(t: np.ndarray, c: np.ndarray) -> float:
    try:
        return float(np.trapezoid(c, t))
    except AttributeError:
        return float(np.trapz(c, t))


def terminal_half_life(t: Sequence[float], c: Sequence[float], n_points: int = 3) -> float:
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    mask = np.isfinite(t) & np.isfinite(c) & (c > 0)
    t, c = t[mask], c[mask]
    if len(t) < n_points:
        return float("nan")
    tt = t[-n_points:]
    cc = c[-n_points:]
    try:
        slope, intercept = np.polyfit(tt, np.log(cc), 1)
    except Exception:
        return float("nan")
    lam = -float(slope)
    return float(np.log(2.0) / lam) if lam > 0 else float("nan")


def nca_metrics(t: Sequence[float], c: Sequence[float], dose_mg: float, route: str, body_weight_g: float) -> Dict[str, float]:
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    mask = np.isfinite(t) & np.isfinite(c)
    t, c = t[mask], c[mask]
    if len(t) < 2:
        return {}
    idx = int(np.argmax(c))
    auc = auc_trapz(t, c)
    dur = float(t[-1] - t[0]) if t[-1] > t[0] else float("nan")
    t_half = terminal_half_life(t, c, n_points=min(5, len(t)))
    if np.isfinite(t_half) and t_half > 0:
        if str(route).lower() == "iv":
            cl = float(dose_mg / auc) if auc > 0 else float("nan")
            vdss = float(cl * t_half / np.log(2.0)) if np.isfinite(cl) else float("nan")
            cl_lbl = "CL"
            vdss_lbl = "Vdss"
        else:
            cl = float(dose_mg / auc) if auc > 0 else float("nan")
            vdss = float(cl * t_half / np.log(2.0)) if np.isfinite(cl) else float("nan")
            cl_lbl = "CL/F"
            vdss_lbl = "Vdss/F"
    else:
        cl = float("nan")
        vdss = float("nan")
        cl_lbl = "CL/F" if str(route).lower() != "iv" else "CL"
        vdss_lbl = "Vdss/F" if str(route).lower() != "iv" else "Vdss"
    return {
        "Cmax": float(c[idx]),
        "Tmax_h": float(t[idx]),
        "AUC0-t": float(auc),
        "Cavg": float(auc / dur) if dur and np.isfinite(dur) and dur > 0 else float("nan"),
        "Clast": float(c[-1]),
        "Tlast_h": float(t[-1]),
        "t1/2_h": float(t_half),
        cl_lbl: float(cl),
        vdss_lbl: float(vdss),
        "BodyWeight_g": float(body_weight_g),
    }


def build_excel_export(df: pd.DataFrame, params: MouseParams, phys: Dict[str, float], kps: Dict[str, float], hep: Dict[str, float], pk: Dict[str, float], ion: Dict[str, object]) -> bytes:
    buf = BytesIO()
    param_df = pd.DataFrame([{"parameter": k, "value": v} for k, v in params.__dict__.items()])
    phys_df = pd.DataFrame([{"item": k, "value": v} for k, v in phys.items()])
    kp_df = pd.DataFrame([{"tissue": k, "Kp": v} for k, v in kps.items()])
    hep_df = pd.DataFrame([{"item": k, "value": v} for k, v in hep.items()])
    pk_df = pd.DataFrame([{"metric": k, "value": v} for k, v in pk.items()])
    ion_df = pd.DataFrame([{"item": k, "value": v} for k, v in ion.items()])

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="simulation")
        param_df.to_excel(writer, index=False, sheet_name="parameters")
        ion_df.to_excel(writer, index=False, sheet_name="ionization")
        phys_df.to_excel(writer, index=False, sheet_name="physiology")
        kp_df.to_excel(writer, index=False, sheet_name="kps")
        hep_df.to_excel(writer, index=False, sheet_name="hepatic_clearance")
        pk_df.to_excel(writer, index=False, sheet_name="pk_summary")
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Sidebar UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Compound inputs")
    example_compound = st.selectbox("Example compound", ["Custom", "Propranolol"], index=1)
    example = PROPRANOLOL_EXAMPLE if example_compound == "Propranolol" else None
    bw_g = st.number_input("Mouse body weight (g)", min_value=10.0, max_value=60.0, value=float(example["bw_g"]) if example else 25.0, step=0.5)
    acidic_pkas_text = st.text_input("Acidic pKa(s) [comma-separated, optional]", value=example["acidic_pkas"] if example else "")
    basic_pkas_text = st.text_input("Basic pKa(s) [comma-separated, optional]", value=example["basic_pkas"] if example else "")
    logp = st.number_input("logP", min_value=-2.0, max_value=8.0, value=float(example["logp"]) if example else 2.5, step=0.1)
    fup = st.number_input("Fraction unbound in plasma (fu/Fup)", min_value=0.0001, max_value=1.0, value=float(example["fup"]) if example else 0.1, step=0.01, format="%.4f")
    rbp = st.number_input("Blood-to-plasma ratio (Rbp)", min_value=0.1, max_value=5.0, value=float(example["rbp"]) if example else 1.5, step=0.1)

    acid_preview = parse_pka_list(acidic_pkas_text)
    base_preview = parse_pka_list(basic_pkas_text)
    ionization_preview = infer_ionization_type(acid_preview, base_preview)
    st.caption(f"Inferred ionization type: **{ionization_preview}**")
    st.caption("Enter pKa(s) as comma-separated values, for example: 4.8 or 4.8, 7.2")

    st.subheader("Microsomal stability")
    tier = st.selectbox("Microsomal t1/2 tier", ["Low (1-30 min)", "Medium (30-60 min)", "High (60-120 min)", "Custom"], index=1 if example_compound == "Propranolol" else 0)
    if tier == "Custom":
        t_half_min = st.number_input("Microsomal t1/2 (min)", min_value=0.1, max_value=500.0, value=45.0, step=1.0)
    else:
        default_t = to_representation_t_half_min(tier)
        if tier == "Low (1-30 min)":
            t_half_min = st.slider("Representative t1/2 (min)", 1.0, 30.0, float(default_t), 0.5)
        elif tier == "Medium (30-60 min)":
            t_half_min = st.slider("Representative t1/2 (min)", 30.0, 60.0, float(default_t), 0.5)
        else:
            t_half_min = st.slider("Representative t1/2 (min)", 60.0, 120.0, float(default_t), 0.5)

    st.subheader("Dosing")
    route = st.selectbox("Route", ["Oral", "IV"], index=0)
    oral_dose_mg = st.number_input("Oral dose (mg)", min_value=0.0, value=float(example["oral_dose_mg"]) if example else 10.0, step=0.5)
    iv_dose_mg = st.number_input("IV dose (mg)", min_value=0.0, value=float(example["iv_dose_mg"]) if example else 5.0, step=0.5)
    dose_interval_h = st.number_input("Dose interval (h)", min_value=0.5, max_value=240.0, value=float(example["dose_interval_h"]) if example else 24.0, step=0.5)
    n_doses = st.number_input("Number of doses", min_value=1, max_value=60, value=int(example["n_doses"]) if example else 1, step=1)
    simulation_days = st.number_input("Simulation horizon (days)", min_value=1.0, max_value=365.0, value=float(example["simulation_days"]) if example else 7.0, step=1.0)

    st.subheader("Observed data")
    obs_file = st.file_uploader("Upload observed CSV", type=["csv"])
    obs_df = None
    obs_time_col = None
    obs_value_col = None
    obs_unit = "ng/mL"
    obs_is_band = False
    if obs_file is not None:
        try:
            obs_df = pd.read_csv(obs_file)
            st.success(f"Loaded {len(obs_df)} rows")
            st.dataframe(obs_df.head(10), use_container_width=True, hide_index=True)
            st.caption("Observed values are interpreted as ng/mL. Convert to ng/mL before upload if needed.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            obs_df = None
    elif example_compound == "Propranolol":
        obs_df = build_propranolol_reference_band(float(simulation_days))
        obs_is_band = True
        st.info("Loaded a default propranolol literature reference band (10–140 ng/mL) for the initial demo view.")
        st.dataframe(obs_df.head(10), use_container_width=True, hide_index=True)

    st.subheader("Display")
    display_series = st.selectbox(
        "Predicted series to plot",
        [
            "Plasma (ng/mL)",
            "Liver (ng/mL)",
            "Brain (ng/mL)",
            "Kidney (ng/mL)",
            "Muscle (ng/mL)",
            "Spleen (ng/mL)",
            "Lung (ng/mL)",
            "Gut (ng/mL)",
            "Adipose (ng/mL)",
            "Bone (ng/mL)",
            "Heart (ng/mL)",
            "Skin (ng/mL)",
            "Rest of body (ng/mL)",
        ],
        index=0,
    )
    y_log = st.checkbox("Log scale", value=False)
    show_table = st.checkbox("Show full simulation table", value=False)

    st.subheader("Observed overlay mapping")
    if obs_df is not None and len(obs_df.columns) > 0:
        num_cols = [c for c in obs_df.columns if pd.api.types.is_numeric_dtype(obs_df[c])]
        if num_cols:
            default_x = num_cols[0] if len(num_cols) >= 1 else obs_df.columns[0]
            default_y = num_cols[1] if len(num_cols) >= 2 else num_cols[0]
            obs_time_col = st.selectbox("Observed time column", obs_df.columns, index=list(obs_df.columns).index(default_x) if default_x in obs_df.columns else 0)
            obs_value_col = st.selectbox("Observed concentration column", obs_df.columns, index=list(obs_df.columns).index(default_y) if default_y in obs_df.columns else 0)
            obs_unit = st.selectbox("Observed units", ["ng/mL", "uM", "nM"], index=0)
        else:
            st.warning("No numeric columns were found in the uploaded CSV.")


# -----------------------------------------------------------------------------
# Build model and simulate
# -----------------------------------------------------------------------------
params = MouseParams(
    bw_g=float(bw_g),
    acidic_pkas=str(acidic_pkas_text),
    basic_pkas=str(basic_pkas_text),
    logp=float(logp),
    fup=float(fup),
    rbp=float(rbp),
    liver_microsome_t_half_min=float(t_half_min),
    route=route,
    oral_dose_mg=float(oral_dose_mg),
    iv_dose_mg=float(iv_dose_mg),
    dose_interval_h=float(dose_interval_h),
    n_doses=int(n_doses),
    simulation_days=float(simulation_days),
)

ion_summary = compound_ionization_summary(params)

with st.spinner("Simulating mouse PBPK model..."):
    df, phys, kps, hep = simulate(params, obs_df)

pk = nca_metrics(df["time_h"].to_numpy(), df[display_series].to_numpy(), dose_mg=float(oral_dose_mg if route == "Oral" else iv_dose_mg), route=route, body_weight_g=float(bw_g))


# -----------------------------------------------------------------------------
# Main tabs
# -----------------------------------------------------------------------------
tab_sim, tab_pk, tab_about = st.tabs(["Simulation", "PK summary", "About"])

with tab_sim:
    left, right = st.columns([2.1, 1.0], gap="large")

    with left:
        st.subheader("Predicted profile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time_h"],
            y=df[display_series],
            mode="lines",
            name=f"Predicted {display_series}",
            hovertemplate="t=%{x:.2f} h<br>Conc=%{y:.3g}<extra></extra>",
        ))

        if obs_df is not None and obs_is_band and {"lower_ng_mL", "upper_ng_mL"}.issubset(obs_df.columns):
            fig.add_trace(go.Scatter(x=obs_df["time_h"], y=obs_df["upper_ng_mL"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=obs_df["time_h"], y=obs_df["lower_ng_mL"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(11,95,255,0.12)", name="Mouse propranolol literature band", hoverinfo="skip"))
        elif obs_df is not None and obs_time_col and obs_value_col:
            obs_plot = obs_df[[obs_time_col, obs_value_col]].dropna().copy()
            obs_y = obs_plot[obs_value_col].astype(float).to_numpy()
            obs_label = f"Observed ({obs_unit})"
            fig.add_trace(go.Scatter(
                x=obs_plot[obs_time_col],
                y=obs_y,
                mode="markers",
                name=obs_label,
                marker=dict(size=8),
                hovertemplate="t=%{x:.2f} h<br>Obs=%{y:.3g}<extra></extra>",
            ))

        fig.update_layout(
            height=540,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Time (h)",
            yaxis_title=display_series,
        )
        if y_log:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Key inputs")
        st.write(
            {
                "BW_g": params.bw_g,
                "acidic_pkas": params.acidic_pkas,
                "basic_pkas": params.basic_pkas,
                "ionization_type": ion_summary["ionization_type"],
                "logP": params.logp,
                "fu": params.fup,
                "Rbp": params.rbp,
                "microsomal_t1/2_min": params.liver_microsome_t_half_min,
                "route": params.route,
                "doses": params.n_doses,
                "interval_h": params.dose_interval_h,
            }
        )
        st.subheader("Dose events")
        dose_events = build_dose_events(params.route, params.oral_dose_mg, params.iv_dose_mg, params.dose_interval_h, params.n_doses)
        st.dataframe(pd.DataFrame(dose_events, columns=["time_h", "dose_mg"]), hide_index=True, use_container_width=True)

        st.subheader("Quick metrics")
        st.metric("Cmax", f"{pk.get('Cmax', float('nan')):.4g}")
        st.metric("Tmax (h)", f"{pk.get('Tmax_h', float('nan')):.4g}")
        st.metric("AUC0-t", f"{pk.get('AUC0-t', float('nan')):.4g}")
        st.metric("Cavg", f"{pk.get('Cavg', float('nan')):.4g}")

with tab_pk:
    st.subheader("PK summary")
    st.dataframe(pd.DataFrame([pk]), use_container_width=True, hide_index=True)

    st.subheader("Ionization inference")
    st.dataframe(pd.DataFrame([ion_summary]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Observed data overlay")
    if obs_df is not None and obs_is_band and {"lower_ng_mL", "upper_ng_mL"}.issubset(obs_df.columns):
        st.dataframe(obs_df.head(20), use_container_width=True, hide_index=True)
        st.caption("Default propranolol example is shown as a literature reference band rather than a digitized concentration-time curve.")
    elif obs_df is not None and obs_time_col and obs_value_col:
        obs_plot = obs_df[[obs_time_col, obs_value_col]].dropna().copy()
        st.dataframe(obs_plot.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Upload a CSV in the sidebar to overlay observed data on the predicted curve.")

    st.divider()
    st.subheader("Kp estimates")
    kp_df = pd.DataFrame([{"tissue": k, "Kp": v} for k, v in kps.items()]).sort_values("tissue")
    st.dataframe(kp_df, use_container_width=True, hide_index=True)

    st.subheader("Microsomal to hepatic clearance conversion")
    hep_df = pd.DataFrame([{"item": k, "value": v} for k, v in hep.items()])
    st.dataframe(hep_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Exports")
    xlsx = build_excel_export(df, params, phys, kps, hep, pk, ion_summary)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download predicted CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="mouse_pbpk_simulation.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Excel workbook",
            data=xlsx,
            file_name="mouse_pbpk_simulation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    if show_table:
        st.divider()
        st.dataframe(df, use_container_width=True)


with tab_about:
    st.subheader("What the app needs from the user")
    st.markdown(
        """
- acidic pKa(s)
- basic pKa(s)
- logP
- fu / Fup
- blood-to-plasma ratio (Rbp)
- microsomal t1/2 category or exact t1/2 value
- dose, route, interval, number of doses
- optional observed CSV for overlay
        """
    )

    st.subheader("Model notes")
    st.markdown(
        """
- Tissue Kp values are generated with a Rowland-Rodgers style expression using fixed mouse tissue composition terms from the provided reference.
- The app infers ionization class from the presence of acidic/basic pKa inputs: neutral, acid, base, or zwitterion.
- Hepatic clearance uses the provided microsomal half-life to in vitro CLint conversion and well-stirred mouse liver scaling.
- The app is intentionally generic and uses only the key compound inputs in the UI.
- The built-in propranolol demo uses a mouse literature reference band (10–140 ng/mL) rather than a digitized full time course; upload your own CSV to replace it.
        """)

