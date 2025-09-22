import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from fuzzy_engine import (
    FuzzyVariableRegistry,
    RuleBase,
    MamdaniEngine,
    ConcernOutputMF,
)

CSV_DIR = os.path.join("membership_function_plots", "csv_data")
RULES_FILE = os.path.join(CSV_DIR, "rules.csv")
CONCERN_OUTPUT_FILE = os.path.join(CSV_DIR, "concern_output_membership_functions.csv")

st.set_page_config(page_title="Fuzzy Concern Estimator", layout="wide")
st.title("Overall Patient Concern")

st.markdown(
    """
    This tool estimates overall concern (0–100%) from vital signs and explains the result in clear, everyday language.
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
    st.stop()

# Load rules and output MF
try:
    concern_mf = ConcernOutputMF.from_csv(CONCERN_OUTPUT_FILE)
    rules = RuleBase.from_csv(RULES_FILE)
    st.subheader("Settings")
    aggregation = st.selectbox(
        "How to combine multiple rules for the same output label",
        options=["max", "prob_sum"],
        index=0,
        help="'max' takes the strongest rule (classic Mamdani). 'prob_sum' softly combines multiple rules: 1 - Π(1 - v).",
    )
    engine = MamdaniEngine(registry, concern_mf, rules, aggregation=aggregation)
except Exception as e:
    st.error(f"Error initializing engine: {e}")
    st.stop()

# ------------------ Presets (render BEFORE inputs so they take effect immediately) ------------------
def norm_key(name: str) -> str:
    return "input__" + "_".join(name.strip().lower().split())

with st.expander("Quick presets"):
    c1, c2, c3, c4 = st.columns(4)

    def set_if(var, val) -> bool:
        """
        Try to set a variable to a value (clipped to its domain).
        Uses the canonical registry name (mf.name) to ensure the widget key matches.
        Returns True if a value was set, False otherwise.
        """
        try:
            mf = registry.get(var)
            vmin, vmax = float(mf.values[0]), float(mf.values[-1])
            clipped = float(np.clip(val, vmin, vmax))
            st.session_state[norm_key(mf.name)] = clipped
            return True
        except Exception:
            return False

    updated_count = 0
    if c1.button("Normal adult"):
        updated_count += int(set_if("Heart Rate", 75))
        updated_count += int(set_if("Respiratory Rate", 16))
        updated_count += int(set_if("Systolic Blood Pressure", 120))
        updated_count += int(set_if("Temperature", 36.8))
        updated_count += int(set_if("Oxygen Saturation", 98))
        updated_count += int(set_if("Inspired Oxygen Concentration", 21))
        updated_count += int(set_if("Supplementary Oxygen", 0))
        st.rerun()

    if c2.button("Hypoxemia on O₂"):
        updated_count = 0
        updated_count += int(set_if("Oxygen Saturation", 88))
        updated_count += int(set_if("Inspired Oxygen Concentration", 35))
        updated_count += int(set_if("Supplementary Oxygen", 6))
        updated_count += int(set_if("Respiratory Rate", 28))
        updated_count += int(set_if("Heart Rate", 110))
        updated_count += int(set_if("Temperature", 37.8))
        updated_count += int(set_if("Systolic Blood Pressure", 115))
        st.rerun()

    if c3.button("Shock-like pattern"):
        updated_count = 0
        updated_count += int(set_if("Systolic Blood Pressure", 85))
        updated_count += int(set_if("Heart Rate", 125))
        updated_count += int(set_if("Respiratory Rate", 26))
        updated_count += int(set_if("Oxygen Saturation", 93))
        updated_count += int(set_if("Inspired Oxygen Concentration", 24))
        updated_count += int(set_if("Supplementary Oxygen", 2))
        updated_count += int(set_if("Temperature", 37.0))
        st.rerun()

    if c4.button("Reset to midpoints"):
        updated_count = 0
        for var in registry.variable_names():
            mf = registry.get(var)
            mid = float(np.nanmedian(mf.values))
            st.session_state[norm_key(mf.name)] = mid
            updated_count += 1
        st.rerun()

# ------------------ Inputs ------------------
st.subheader("Enter the vital signs")
cols = st.columns(2)
for i, var in enumerate(registry.variable_names()):
    mf = registry.get(var)
    vmin, vmax = float(mf.values[0]), float(mf.values[-1])
    default = float(np.clip(np.nanmedian(mf.values), vmin, vmax))
    key = norm_key(var)
    if key not in st.session_state:
        st.session_state[key] = default
    with cols[i % 2]:
        st.number_input(var, min_value=vmin, max_value=vmax, value=st.session_state[key], step=(vmax - vmin) / 100.0, key=key)

# Build dict
inputs: Dict[str, float] = {var: float(st.session_state[norm_key(var)]) for var in registry.variable_names()}

# ------------------ Run ------------------
if st.button("Evaluate"):
    with st.spinner("Running Mamdani inference..."):
        result = engine.evaluate(inputs)

    # Display headline result (scaled to percent if needed)
    vmin, vmax = result["output_domain"]
    concern_pct = float(result["concern_percent"])
    # Pick descriptor based on percent scale
    if concern_pct < 20:
        desc, color = "very low", "#1b5e20"
    elif concern_pct < 40:
        desc, color = "low", "#2e7d32"
    elif concern_pct < 60:
        desc, color = "moderate", "#f9a825"
    elif concern_pct < 80:
        desc, color = "high", "#ef6c00"
    else:
        desc, color = "very high", "#b71c1c"

    st.markdown(
        f"""
        <div style='padding:1rem;border-radius:8px;background:{color}20;border:1px solid {color};'>
            <span style='font-size:1.1rem;'>Overall concern is</span>
            <span style='font-size:1.2rem;font-weight:700;'> {concern_pct:.1f}%</span>
            <span style='opacity:0.9'>(~{desc}).</span>
            <div style='opacity:0.7;font-size:0.9rem;'>Calculated from the output scale [{vmin:g}, {vmax:g}].</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("How this score was calculated")

    # Input interpretations in simple language (non-technical)
    st.markdown("**What each number suggests**")
    input_mems = result["input_memberships"]

    def friendly(label: str) -> Tuple[str, str]:
        ll = label.lower()
        if ll == "no concern":
            return ("Normal", "#2e7d32")
        if "severe" in ll:
            return ("Very abnormal", "#b71c1c")
        if "moderate" in ll:
            return ("Abnormal", "#ef6c00")
        if "mild" in ll:
            return ("Slightly abnormal", "#f9a825")
        if "above normal" in ll:
            return ("High", "#ef6c00")
        if "below normal" in ll:
            return ("Low", "#ef6c00")
        return (label.title(), "#607d8b")

    lines = []
    for var_name, x in inputs.items():
        mems = input_mems[var_name]
        # choose the single best-matching label
        best_label, best_deg = max(mems.items(), key=lambda kv: kv[1])
        phrase, color = friendly(best_label)
        line = (
            f"<div style='margin:4px 0;'>"
            f"<strong>{var_name}</strong> at <strong>{x:g}</strong>: "
            f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;background:{color}20;border:1px solid {color};color:{color};'>"
            f"{phrase}"
            f"</span>"
            f"</div>"
        )
        lines.append(line)
    st.markdown("\n".join(lines), unsafe_allow_html=True)

    # Simple narrative instead of rules list
    st.markdown("**Top signals influencing the score**")
    rule_firings = result["rule_firings"]
    # Exclude rules whose consequent is the default 'No concern'
    rule_firings_non_default = [
        r for r in rule_firings
        if str(r.get("ConsequentLabel", "")).strip().lower() != "no concern"
    ]
    if rule_firings_non_default:
        # Gather strongest antecedent signals from the strongest fired rules
        candidates = []  # (degree, variable, label)
        for r in sorted(rule_firings_non_default, key=lambda rr: rr.get("FiringStrength", 0.0), reverse=True)[:7]:
            if r.get("FiringStrength", 0.0) <= 0:
                continue
            for a in r.get("Antecedents", []):
                candidates.append((float(a.get("degree", 0.0)), str(a.get("variable", "")), str(a.get("label", ""))))
        candidates.sort(reverse=True)
        # Deduplicate by variable, keep top 3
        picked = []
        seen_vars = set()
        for deg, var, lab in candidates:
            if var in seen_vars:
                continue
            seen_vars.add(var)
            picked.append((var, lab, deg))
            if len(picked) == 3:
                break
        if picked:
            lines = []
            for var, lab, deg in picked:
                lines.append(f"- {var}: best matches '{lab}' (confidence ~{deg:.2f}).")
            st.markdown("\n".join(lines))
        else:
            st.markdown("- No strong risk signals were detected.")
    else:
        st.markdown("- No strong risk signals were detected.")

    # Output label activity (grouped influence)
    st.markdown("**Which concern levels were most active**")
    agg = result["aggregated_output"]
    label_peaks = []
    for label, vec in agg.items():
        v = np.array(vec, dtype=float)
        peak = float(np.max(v)) if v.size else 0.0
        label_peaks.append((label, peak))
    label_peaks.sort(key=lambda x: x[1], reverse=True)
    if label_peaks and max(p for _, p in label_peaks) > 0:
        df_peaks = pd.DataFrame(label_peaks, columns=["Concern level", "Peak activation"]).set_index("Concern level")
        st.bar_chart(df_peaks)
    else:
        st.markdown("- No output labels were activated.")

    # Drivers: what raised vs lowered the concern
    st.markdown("**What pushed the score up or down**")
    inc_bullets, dec_bullets = [], []
    # Increasing drivers from fired rules' antecedents
    for r in rule_firings_non_default[:10]:
        if r.get("FiringStrength", 0.0) <= 0:
            continue
        for a in r.get("Antecedents", []):
            label = str(a.get("label", ""))
            var = str(a.get("variable", ""))
            deg = float(a.get("degree", 0.0))
            if any(key in label.lower() for key in ["above normal", "below normal", "severe", "moderate", "mild"]):
                inc_bullets.append(f"- {var}: '{label}' (~{deg:.2f})")
    # Decreasing drivers (reassuring) from inputs with high 'No concern'
    for var_name, mems in input_mems.items():
        nc = mems.get("No concern", 0.0)
        if nc >= 0.7:
            dec_bullets.append(f"- {var_name}: 'No concern' is strong (~{nc:.2f})")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Things raising concern**")
        st.markdown("\n".join(inc_bullets[:6]) or "- None prominent.")
    with cols[1]:
        st.markdown("**Things lowering concern**")
        st.markdown("\n".join(dec_bullets[:6]) or "- None prominent.")

    # (Optional diagnostics removed to keep this app minimal and error-free.)

    # Gentle diagnostic: why some inputs may not change the score much
    with st.expander("If a number seems to have little effect"):
        st.markdown(
            "- If a value is already near 'Normal', moving it a little may not change the score much.\n"
            "- A single reading can matter more when combined with others (for example, high oxygen needs plus low oxygen saturation).\n"
            "- Values outside the charted range are clipped to the nearest allowed value.\n"
        )

        # Show current dominant label and whether any high-concern rules reference it
        rows_diag = []
        for var_name in registry.variable_names():
            mems = input_mems.get(var_name, {})
            if not mems:
                continue
            best_label, best_deg = max(mems.items(), key=lambda kv: kv[1])
            # scan rule base for any rule that includes (var_name, best_label)
            has_strong = False
            max_weight = 0.0
            target_conseq = None
            try:
                for r in rules.rules:
                    if any((a == var_name or a.lower() == var_name.lower()) and b == best_label for (a, b) in r.antecedent):
                        max_weight = max(max_weight, float(getattr(r, 'weight', 1.0)))
                        if r.consequent_label.lower() in {"high", "very high"}:
                            has_strong = True
                            target_conseq = r.consequent_label
            except Exception:
                pass
            rows_diag.append({
                "Variable": var_name,
                "Best match": best_label,
                "Confidence": f"{best_deg:.2f}",
                "Linked to high-concern rule?": "Yes" if has_strong else "No",
                "Rule importance": f"{max_weight:.2f}",
            })
        st.dataframe(pd.DataFrame(rows_diag), use_container_width=True)

        # Quick local sensitivity: nudge each input by a small step and show Δ in concern percent
        def _evaluate_with(inputs_override: Dict[str, float]) -> float:
            res2 = engine.evaluate(inputs_override)
            return float(res2.get("concern_percent", 0.0))

        step_rows = []
        base_pct = concern_pct
        for var_name in registry.variable_names():
            mf = registry.get(var_name)
            vmin, vmax = float(mf.values[0]), float(mf.values[-1])
            span = max(1e-9, vmax - vmin)
            step = 0.05 * span  # 5% of range
            base_val = inputs[var_name]
            up_val = float(np.clip(base_val + step, vmin, vmax))
            down_val = float(np.clip(base_val - step, vmin, vmax))
            test_up = dict(inputs)
            test_down = dict(inputs)
            test_up[var_name] = up_val
            test_down[var_name] = down_val
            up_pct = _evaluate_with(test_up)
            down_pct = _evaluate_with(test_down)
            step_rows.append({
                "Variable": var_name,
                "Δ if nudged up": f"{(up_pct - base_pct):+.2f} pts",
                "Δ if nudged down": f"{(down_pct - base_pct):+.2f} pts",
            })
        st.markdown("**Small changes, small effects** — estimated impact of a small nudge up or down:")
        st.dataframe(pd.DataFrame(step_rows), use_container_width=True)

st.caption("Notes: values outside the supported range are clipped; the percentage simply rescales the output to 0–100.")
