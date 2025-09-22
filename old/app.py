import os
import sys
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from fuzzy_engine import (
    MembershipFunctionSet,
    FuzzyVariableRegistry,
    Rule,
    RuleBase,
    MamdaniEngine,
    ConcernOutputMF,
)

CSV_DIR = os.path.join("membership_function_plots", "csv_data")
RULES_FILE = os.path.join(CSV_DIR, "rules.csv")
CONCERN_OUTPUT_FILE = os.path.join(CSV_DIR, "concern_output_membership_functions.csv")

st.set_page_config(page_title="Fuzzy Concern Estimator", layout="wide")
st.title("Patient Concern Level (Mamdani Fuzzy Logic)")

st.markdown(
    """
    This tool estimates an overall level of concern (0–100%) from vital signs using a Mamdani fuzzy logic system.
    It uses your exact membership function data as provided (no simplification), and explains its reasoning in
    clear, human-friendly language.
    """
)

# Load input membership functions from your CSVs
with st.spinner("Loading membership functions..."):
    try:
        registry = FuzzyVariableRegistry.from_directory(CSV_DIR)
    except Exception as e:
        st.error(f"Failed to load membership functions from {CSV_DIR}: {e}")
        st.stop()

# Check for rules and output MF
missing: List[str] = []
if not os.path.exists(RULES_FILE):
    missing.append(f"Rules file not found: `{RULES_FILE}`")
if not os.path.exists(CONCERN_OUTPUT_FILE):
    missing.append(f"Concern output MF file not found: `{CONCERN_OUTPUT_FILE}`")

if missing:
    st.error("\n".join(missing))
    st.info(
        """
        To run the inference, please provide:
        1) A rules CSV at `membership_function_plots/csv_data/rules.csv`
        2) An output membership function CSV at `membership_function_plots/csv_data/concern_output_membership_functions.csv`

        Expected rules format (CSV):
        - Columns: `RuleID`, `Logic`, `Antecedent` (JSON array), `ConsequentLabel`
        - `Logic` is either `AND` or `OR` (defines how antecedents combine)
        - `Antecedent` is a JSON array of objects: [{"variable": "Heart rate", "label": "Above normal - severe concern"}, ...]
        - `ConsequentLabel` must match a column name in the concern output MF CSV.

        Example Antecedent JSON: `[{"variable": "oxygen saturation", "label": "Below normal - moderate concern"}, {"variable": "supplementary oxygen", "label": "Yes"}]`

        Expected concern output MF CSV format:
        - Columns: `Value`, plus one column per output label (e.g., `Very low`, `Low`, `Moderate`, `High`, `Very high`)
        - `Value` should span 0..100 (or your exact universe of discourse), with precise membership values per label (no simplification).
        """
    )
else:
    # Load rules and output MF
    try:
        concern_mf = ConcernOutputMF.from_csv(CONCERN_OUTPUT_FILE)
        rules = RuleBase.from_csv(RULES_FILE)
        st.subheader("Model settings")
        aggregation = st.selectbox(
            "How to combine multiple rules for the same output label",
            options=["max", "prob_sum"],
            index=0,
            help="'max' takes the strongest rule (classic Mamdani). 'prob_sum' softly combines multiple rules: 1 - Π(1 - v).",
        )
        # Optional fix for obviously inverted oxygen-related CSVs
        st.markdown("\n")
        with st.expander("Data sanity checks and quick fixes"):
            st.caption("If your Inspired/Supplementary Oxygen membership CSVs were exported with inverted columns, enable this fix.")

            def _shift_oxygen_columns_left(mfset):
                expected = [
                    "No concern",
                    "Above normal - mild concern",
                    "Above normal - moderate concern",
                    "Above normal - severe concern",
                ]
                # Verify all expected labels are present
                if all(lbl in mfset.labels for lbl in expected):
                    # Reorder columns by performing a circular left shift so that the column that is 1.0 at normal becomes "No concern"
                    # Build index map in expected order
                    idxs = [mfset.labels.index(lbl) for lbl in expected]
                    sub = mfset.matrix[:, idxs]
                    # Circular left shift by 1
                    shifted = np.concatenate([sub[:, 1:], sub[:, :1]], axis=1)
                    # Write back
                    for i, lbl in enumerate(expected):
                        mfset.matrix[:, idxs[i]] = shifted[:, i]
                return mfset

            def _detect_inversion(mfset):
                # Heuristic: if at the minimum Value, "No concern" is very low while one of the "Above normal" labels is very high,
                # and at the maximum Value the reverse is true, flag as inverted.
                try:
                    vals = mfset.values
                    labels = mfset.labels
                    mat = mfset.matrix
                    if "No concern" not in labels:
                        return False
                    no_idx = labels.index("No concern")
                    top_labels = [l for l in labels if l != "No concern" and ("Above normal" in l or "concern" in l)]
                    if not top_labels:
                        return False
                    top_idxs = [labels.index(l) for l in top_labels]
                    at_min_no = float(mat[0, no_idx])
                    at_min_top = float(np.max(mat[0, top_idxs]))
                    at_max_no = float(mat[-1, no_idx])
                    at_max_top = float(np.max(mat[-1, top_idxs]))
                    return (at_min_no < 0.2 and at_min_top > 0.8 and at_max_no > 0.8 and at_max_top < 0.2)
                except Exception:
                    return False

            fix_oxy = st.checkbox("Apply fix for inverted Inspired/Supplementary Oxygen membership columns", value=True)
            oxy_vars = [
                "Inspired Oxygen Concentration",
                "Supplementary Oxygen",
            ]
            inversion_flags = {}
            for v in oxy_vars:
                try:
                    mfv = registry.get(v)
                    inversion_flags[v] = _detect_inversion(mfv)
                except Exception:
                    inversion_flags[v] = False
            if any(inversion_flags.values()):
                st.warning(
                    "Detected likely inversion in: "
                    + ", ".join([k for k, f in inversion_flags.items() if f])
                    + ". Enable the fix below or correct the CSVs."
                )
            if fix_oxy:
                for v in oxy_vars:
                    try:
                        mfv = registry.get(v)
                        _shift_oxygen_columns_left(mfv)
                    except Exception:
                        pass

        engine = MamdaniEngine(registry, concern_mf, rules, aggregation=aggregation)

        # Optional: quick clinical presets BEFORE inputs so changes show immediately in this run
        def norm_key(name: str) -> str:
            return "input__" + "_".join(name.strip().lower().split())

        with st.expander("Quick test presets"):
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Normal adult"):
                def set_if(var, val):
                    try:
                        mfset = registry.get(var)
                        clipped = float(np.clip(val, mfset.values.min(), mfset.values.max()))
                        st.session_state[norm_key(var)] = clipped
                    except Exception:
                        pass
                set_if("Heart Rate", 75)
                set_if("Respiratory Rate", 16)
                set_if("Systolic Blood Pressure", 120)
                set_if("Temperature", 36.8)
                set_if("Oxygen Saturation", 98)
                set_if("Inspired Oxygen Concentration", 21)
                set_if("Supplementary Oxygen", 0)
                st.rerun()
            if c2.button("Hypoxemia on O2"):
                def set_if(var, val):
                    try:
                        mfset = registry.get(var)
                        clipped = float(np.clip(val, mfset.values.min(), mfset.values.max()))
                        st.session_state[norm_key(var)] = clipped
                    except Exception:
                        pass
                set_if("Oxygen Saturation", 88)
                set_if("Inspired Oxygen Concentration", 35)
                set_if("Supplementary Oxygen", 6)
                set_if("Respiratory Rate", 28)
                set_if("Heart Rate", 110)
                set_if("Temperature", 37.8)
                set_if("Systolic Blood Pressure", 115)
                st.rerun()
            if c3.button("Shock-like pattern"):
                def set_if(var, val):
                    try:
                        mfset = registry.get(var)
                        clipped = float(np.clip(val, mfset.values.min(), mfset.values.max()))
                        st.session_state[norm_key(var)] = clipped
                    except Exception:
                        pass
                set_if("Systolic Blood Pressure", 85)
                set_if("Heart Rate", 125)
                set_if("Respiratory Rate", 26)
                set_if("Oxygen Saturation", 93)
                set_if("Inspired Oxygen Concentration", 24)
                set_if("Supplementary Oxygen", 2)
                set_if("Temperature", 37.0)
                st.rerun()
            if c4.button("Reset defaults"):
                for var in registry.variable_names():
                    mfset = registry.get(var)
                    default = float(np.clip(np.nanmean(mfset.values), mfset.values.min(), mfset.values.max()))
                    st.session_state[norm_key(var)] = default
                st.rerun()
    except Exception as e:
        st.error(f"Error initializing engine: {e}")
        st.stop()

    # Build input UI dynamically from loaded variables
    st.subheader("Enter the vital signs")

    # Render inputs and keep values in session_state so presets can update reliably
    cols = st.columns(2)
    idx = 0
    for var in registry.variable_names():
        mfset = registry.get(var)
        vmin, vmax = float(mfset.values.min()), float(mfset.values.max())
        default = float(np.clip(np.nanmean(mfset.values), vmin, vmax))
        key = norm_key(var)
        if key not in st.session_state:
            st.session_state[key] = default
        with cols[idx % 2]:
            st.number_input(var, min_value=vmin, max_value=vmax, value=st.session_state[key], step=(vmax - vmin) / 100.0, key=key)
        idx += 1

    # Build inputs from session state
    inputs: Dict[str, float] = {}
    for var in registry.variable_names():
        key = norm_key(var)
        inputs[var] = float(st.session_state.get(key))

    # (Presets are now rendered before inputs; removed duplicates here.)

    if st.button("Evaluate"):
        with st.spinner("Running Mamdani inference..."):
            result = engine.evaluate(inputs)

        # 1) Overall concern with a qualitative descriptor
        concern_value = float(result['concern_value'])
        if concern_value < 20:
            desc = "very low"
            color = "#1b5e20"
        elif concern_value < 40:
            desc = "low"
            color = "#2e7d32"
        elif concern_value < 60:
            desc = "moderate"
            color = "#f9a825"
        elif concern_value < 80:
            desc = "high"
            color = "#ef6c00"
        else:
            desc = "very high"
            color = "#b71c1c"

        st.markdown(
            f"""
            <div style='padding:1rem;border-radius:8px;background:{color}20;border:1px solid {color};'>
                <span style='font-size:1.1rem;'>Overall concern is</span>
                <span style='font-size:1.2rem;font-weight:700;'> {concern_value:.2f}%</span>
                <span style='opacity:0.9'>(approximately {desc}).</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("How the decision was formed")

        # 2) Explain strongest memberships for each input (top 1–2 per variable)
        st.markdown("**Input interpretations**")
        input_mems = result["input_memberships"]
        bullets = []
        for var_name, x in inputs.items():
            mems = input_mems[var_name]
            # sort labels by degree
            top = sorted(mems.items(), key=lambda kv: kv[1], reverse=True)
            # keep top memberships above a small threshold
            top_filtered = [(l, d) for l, d in top[:3] if d >= 0.05]
            if not top_filtered and top:
                top_filtered = top[:1]
            parts = ", ".join([f"{label} ({degree:.2f})" for label, degree in top_filtered])
            bullets.append(f"- **{var_name}** at {x:g} is most consistent with: {parts}.")
        st.markdown("\n".join(bullets))

        # 3) Explain which rules fired the most (top 5)
        st.markdown("**Key rules that influenced the result**")
        rule_firings = result["rule_firings"]
        if rule_firings:
            # sort by firing strength
            top_rules = sorted(rule_firings, key=lambda r: r.get("FiringStrength", 0.0), reverse=True)
            top_rules = [r for r in top_rules if r.get("FiringStrength", 0.0) > 0][:5]
            if top_rules:
                explanations = []
                for r in top_rules:
                    ants = ", ".join([
                        f"{a['variable']} is '{a['label']}' ({a['degree']:.2f})" for a in r.get("Antecedents", [])
                    ])
                    explanations.append(
                        f"- Rule {r.get('RuleID') or '—'} ({r.get('Logic')}): IF {ants} THEN concern is '{r.get('ConsequentLabel')}'. Fired at {r.get('FiringStrength', 0.0):.2f}."
                    )
                st.markdown("\n".join(explanations))
            else:
                st.markdown("- No rules fired strongly for this input set.")
        else:
            st.markdown("- No rules fired.")

        # 4) Summarize which output labels were most active
        st.markdown("**Which concern levels were most active**")
        agg = result["aggregated_output"]  # dict label -> list
        label_peaks = []
        for label, vec in agg.items():
            v = np.array(vec, dtype=float)
            peak = float(np.max(v)) if v.size else 0.0
            label_peaks.append((label, peak))
        label_peaks.sort(key=lambda x: x[1], reverse=True)
        desc_lines = [f"- '{label}' reached an activation of {peak:.2f}." for label, peak in label_peaks if peak > 0]
        if desc_lines:
            st.markdown("\n".join(desc_lines))
        else:
            st.markdown("- No output labels were activated.")

st.caption("Note: This app strictly uses your provided membership function CSVs without any shape approximation.")
