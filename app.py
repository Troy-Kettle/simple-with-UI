import os
from typing import Dict, List, Tuple

import altair as alt
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
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        aggregation = st.selectbox(
            "How to combine multiple rules for the same output label",
            options=["max", "prob_sum"],
            index=0,
            help="'max' takes the strongest rule (classic Mamdani). 'prob_sum' softly combines multiple rules: 1 - Π(1 - v).",
        )
    with col_set2:
        normalize_mfs = st.checkbox(
            "Normalise MFs to [0,1]",
            value=False,
            help="Scale all fuzzy membership functions so that each label reaches a maximum value of 1.0. Useful when some membership functions don't reach full activation."
        )
    
    # Apply normalisation to registry if enabled
    active_registry = registry.normalize_all() if normalize_mfs else registry
    
    # Show normalisation info if enabled
    if normalize_mfs:
        stats_before = registry.stats()
        mfs_needing_norm = [var for var, stat in stats_before.items() if stat['min_max'] < 0.99]
        if mfs_needing_norm:
            st.info(f"Normalising {len(mfs_needing_norm)} membership function(s): {', '.join(mfs_needing_norm)}")
    
    engine = MamdaniEngine(active_registry, concern_mf, rules, aggregation=aggregation)
except Exception as e:
    st.error(f"Error initialising engine: {e}")
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
def calculate_news2(inputs: Dict[str, float]) -> Tuple[int, Dict[str, int], str]:
    """
    Calculate NEWS2 score from vital signs.
    Returns: (total_score, score_breakdown, clinical_risk_level)
    """
    scores = {}
    
    # Respiratory Rate
    rr = inputs.get("Respiratory Rate", 0)
    if rr <= 8:
        scores["Respiratory Rate"] = 3
    elif rr <= 11:
        scores["Respiratory Rate"] = 1
    elif rr <= 20:
        scores["Respiratory Rate"] = 0
    elif rr <= 24:
        scores["Respiratory Rate"] = 2
    else:
        scores["Respiratory Rate"] = 3
    
    # SpO2 (Scale 1)
    spo2 = inputs.get("Oxygen Saturation", 0)
    if spo2 <= 91:
        scores["Oxygen Saturation"] = 3
    elif spo2 <= 93:
        scores["Oxygen Saturation"] = 2
    elif spo2 <= 95:
        scores["Oxygen Saturation"] = 1
    else:
        scores["Oxygen Saturation"] = 0
    
    # Supplemental Oxygen
    supp_o2 = inputs.get("Supplementary Oxygen", 0)
    scores["Supplementary Oxygen"] = 2 if supp_o2 > 0 else 0
    
    # Systolic BP
    sbp = inputs.get("Systolic Blood Pressure", 0)
    if sbp <= 90:
        scores["Systolic Blood Pressure"] = 3
    elif sbp <= 100:
        scores["Systolic Blood Pressure"] = 2
    elif sbp <= 110:
        scores["Systolic Blood Pressure"] = 1
    elif sbp <= 219:
        scores["Systolic Blood Pressure"] = 0
    else:
        scores["Systolic Blood Pressure"] = 3
    
    # Heart Rate (Pulse)
    hr = inputs.get("Heart Rate", 0)
    if hr <= 40:
        scores["Heart Rate"] = 3
    elif hr <= 50:
        scores["Heart Rate"] = 1
    elif hr <= 90:
        scores["Heart Rate"] = 0
    elif hr <= 110:
        scores["Heart Rate"] = 1
    elif hr <= 130:
        scores["Heart Rate"] = 2
    else:
        scores["Heart Rate"] = 3
    
    # Temperature
    temp = inputs.get("Temperature", 0)
    if temp <= 35.0:
        scores["Temperature"] = 3
    elif temp <= 36.0:
        scores["Temperature"] = 1
    elif temp <= 38.0:
        scores["Temperature"] = 0
    elif temp <= 39.0:
        scores["Temperature"] = 1
    else:
        scores["Temperature"] = 2
    
    total = sum(scores.values())
    
    # Clinical risk
    if total == 0:
        risk = "Low"
    elif total <= 4:
        risk = "Low"
    elif total <= 6:
        risk = "Medium"
    else:
        risk = "High"
    
    return total, scores, risk

# Initialize session state for evaluation results
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

if st.button("Evaluate"):
    with st.spinner("Running Mamdani inference..."):
        result = engine.evaluate(inputs)

    # Calculate NEWS2
    news2_score, news2_breakdown, news2_risk = calculate_news2(inputs)

    # Save to session state
    st.session_state.evaluation_result = {
        "result": result,
        "news2_score": news2_score,
        "news2_breakdown": news2_breakdown,
        "news2_risk": news2_risk,
        "inputs": inputs.copy(),
    }
    # Clear Monte Carlo results when new evaluation is completed
    st.session_state.mc_results = None
    st.session_state.mc_stats = None

# Display results if available
if st.session_state.evaluation_result is not None:
    eval_data = st.session_state.evaluation_result
    result = eval_data["result"]
    news2_score = eval_data["news2_score"]
    news2_breakdown = eval_data["news2_breakdown"]
    news2_risk = eval_data["news2_risk"]
    inputs = eval_data["inputs"]

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

    settings_html = "<div style='opacity:0.7;font-size:0.85rem;margin-top:0.3rem;'>MF normalisation enabled</div>" if normalize_mfs else ""
    
    st.markdown(
        f"""
        <div style='padding:1rem;border-radius:8px;background:{color}20;border:1px solid {color};'>
            <span style='font-size:1.1rem;'>Overall concern is</span>
            <span style='font-size:1.2rem;font-weight:700;'> {concern_pct:.1f}%</span>
            <span style='opacity:0.9'>(~{desc}).</span>
            <div style='opacity:0.7;font-size:0.9rem;'>Calculated from the output scale [{vmin:g}, {vmax:g}].</div>
            {settings_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Fuzzy Logic Interpretation", "NEWS2 Comparison", "Uncertainty Analysis (Monte Carlo)"])
    
    with tab1:
        st.subheader("How this score was calculated")

        # Input interpretations in simple language (non-technical)
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

        # Get rule firings for top signals
        rule_firings = result["rule_firings"]
        rule_firings_non_default = [
            r for r in rule_firings
            if str(r.get("ConsequentLabel", "")).strip().lower() != "no concern"
        ]
        
        # Gather strongest antecedent signals from the strongest fired rules
        candidates = []  # (degree, variable, label)
        for r in sorted(rule_firings_non_default, key=lambda rr: rr.get("FiringStrength", 0.0), reverse=True)[:7]:
            if r.get("FiringStrength", 0.0) <= 0:
                continue
            for a in r.get("Antecedents", []):
                candidates.append((float(a.get("degree", 0.0)), str(a.get("variable", "")), str(a.get("label", ""))))
        candidates.sort(reverse=True)
        
        # Deduplicate by variable
        top_signals_dict = {}
        for deg, var, lab in candidates:
            if var not in top_signals_dict:
                top_signals_dict[var] = (lab, deg)
        
        # Collect raising and lowering factors
        inc_factors, dec_factors = {}, {}
        for r in rule_firings_non_default[:10]:
            if r.get("FiringStrength", 0.0) <= 0:
                continue
            for a in r.get("Antecedents", []):
                label = str(a.get("label", ""))
                var = str(a.get("variable", ""))
                deg = float(a.get("degree", 0.0))
                if any(key in label.lower() for key in ["above normal", "below normal", "severe", "moderate", "mild"]):
                    if var not in inc_factors or deg > inc_factors[var][1]:
                        inc_factors[var] = (label, deg)
        
        for var_name, mems in input_mems.items():
            nc = mems.get("No concern", 0.0)
            if nc >= 0.7:
                dec_factors[var_name] = ("No concern", nc)
        
        # Build interpretability table
        table_rows = []
        for var_name in registry.variable_names():
            x = inputs[var_name]
            mems = input_mems[var_name]
            
            # What each number suggests
            best_label, best_deg = max(mems.items(), key=lambda kv: kv[1])
            phrase, _ = friendly(best_label)
            what_suggests = f"{phrase} ({x:g})"
            
            # Top signals influencing
            if var_name in top_signals_dict:
                sig_label, sig_deg = top_signals_dict[var_name]
                top_signal = f"{sig_label} (~{sig_deg:.2f})"
            else:
                top_signal = "—"
            
            # Raising/Lowering concern
            concern_change = []
            if var_name in inc_factors:
                inc_label, inc_deg = inc_factors[var_name]
                concern_change.append(f"↑ {inc_label} ({inc_deg:.2f})")
            if var_name in dec_factors:
                dec_label, dec_deg = dec_factors[var_name]
                concern_change.append(f"↓ {dec_label} ({dec_deg:.2f})")
            concern_str = ", ".join(concern_change) if concern_change else "—"
            
            table_rows.append({
                "Variable": var_name,
                "What Each Number Suggests": what_suggests,
                "Top Signals Influencing": top_signal,
                "Raising/Lowering Concern": concern_str
            })
        
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("NEWS2 vs Fuzzy Logic Comparison")
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### NEWS2 Score")
            # Colour code NEWS2 based on risk
            if news2_risk == "Low":
                news2_color = "#2e7d32"
            elif news2_risk == "Medium":
                news2_color = "#f9a825"
            else:
                news2_color = "#b71c1c"
            
            st.markdown(
                f"""
                <div style='padding:1rem;border-radius:8px;background:{news2_color}20;border:1px solid {news2_color};'>
                    <span style='font-size:1.1rem;'>NEWS2 Score:</span>
                    <span style='font-size:1.2rem;font-weight:700;'> {news2_score}</span>
                    <div style='opacity:0.9;margin-top:0.5rem;'>Clinical Risk: <strong>{news2_risk}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown("**Score Breakdown:**")
            breakdown_df = pd.DataFrame([
                {"Parameter": k, "Value": inputs.get(k, "N/A"), "Points": v}
                for k, v in news2_breakdown.items()
            ])
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            st.info("**Note:** NEWS2 does not include consciousness level (AVPU) in this calculation, as it is not part of the input vitals.")
        
        with col2:
            st.markdown("### Fuzzy Logic Score")
            st.markdown(
                f"""
                <div style='padding:1rem;border-radius:8px;background:{color}20;border:1px solid {color};'>
                    <span style='font-size:1.1rem;'>Concern:</span>
                    <span style='font-size:1.2rem;font-weight:700;'> {concern_pct:.1f}%</span>
                    <div style='opacity:0.9;margin-top:0.5rem;'>Level: <strong>{desc.title()}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    with tab3:
        st.subheader("Uncertainty Analysis via Monte Carlo Simulation")
        
        st.markdown("""
        This analysis quantifies how **measurement uncertainty** in vital signs affects the concern score.
        Each input is perturbed according to typical clinical measurement error, and the fuzzy inference
        is run multiple times to generate a distribution of possible outputs.
        """)
        
        # Default measurement uncertainties (standard deviations)
        default_uncertainties = {
            "Heart Rate": 2.5,
            "Respiratory Rate": 1.5,
            "Systolic Blood Pressure": 5.0,
            "Temperature": 0.2,
            "Oxygen Saturation": 2.0,
            "Inspired Oxygen Concentration": 2.0,
            "Supplementary Oxygen": 0.5,
        }
        
        col_mc1, col_mc2 = st.columns([2, 1])
        
        with col_mc1:
            n_samples = st.slider("Number of Monte Carlo samples", 100, 2000, 500, step=100,
                                  help="More samples = more accurate uncertainty estimates but slower computation")
        
        with col_mc2:
            show_uncertainties = st.checkbox("Customise uncertainties", value=False)
        
        # Allow user to customise uncertainties if desired
        uncertainties = default_uncertainties.copy()
        if show_uncertainties:
            st.markdown("**Measurement uncertainties (± standard deviation):**")
            unc_cols = st.columns(3)
            for idx, (var, default_unc) in enumerate(default_uncertainties.items()):
                with unc_cols[idx % 3]:
                    uncertainties[var] = st.number_input(
                        f"{var}",
                        min_value=0.0,
                        value=default_unc,
                        step=0.1,
                        format="%.2f",
                        key=f"unc_{var}"
                    )
        
        # Initialize session state for MC results
        if "mc_results" not in st.session_state:
            st.session_state.mc_results = None
            st.session_state.mc_stats = None
        
        if st.button("Run Monte Carlo Analysis", key="run_mc"):
            with st.spinner(f"Running {n_samples} simulations..."):
                # Run Monte Carlo
                mc_results = []
                
                for _ in range(n_samples):
                    # Perturb inputs
                    perturbed_inputs = {}
                    for var_name, value in inputs.items():
                        mf = registry.get(var_name)
                        vmin, vmax = float(mf.values[0]), float(mf.values[-1])
                        
                        # Sample from normal distribution
                        unc = uncertainties.get(var_name, 0.0)
                        perturbed_value = np.random.normal(value, unc)
                        
                        # Clip to valid range
                        perturbed_value = float(np.clip(perturbed_value, vmin, vmax))
                        perturbed_inputs[var_name] = perturbed_value
                    
                    # Evaluate with perturbed inputs
                    mc_result = engine.evaluate(perturbed_inputs)
                    mc_results.append(mc_result["concern_percent"])
                
                mc_results = np.array(mc_results)
                
                # Statistics
                mean_concern = np.mean(mc_results)
                std_concern = np.std(mc_results)
                ci_lower = np.percentile(mc_results, 2.5)
                ci_upper = np.percentile(mc_results, 97.5)
                
                # Save to session state
                st.session_state.mc_results = mc_results
                st.session_state.mc_stats = {
                    "mean": mean_concern,
                    "std": std_concern,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "base_concern": concern_pct,
                    "uncertainties": uncertainties.copy(),
                    "inputs": inputs.copy()
                }
        
        # Display results if available
        if st.session_state.mc_results is not None:
            mc_results = st.session_state.mc_results
            stats = st.session_state.mc_stats
            mean_concern = stats["mean"]
            std_concern = stats["std"]
            ci_lower = stats["ci_lower"]
            ci_upper = stats["ci_upper"]
            base_concern_mc = stats["base_concern"]
            
            # Display results
            st.markdown("---")
            st.markdown("### Results")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Mean Concern", f"{mean_concern:.1f}%", 
                         delta=f"{mean_concern - base_concern_mc:+.1f}% from base")
            
            with col_r2:
                st.metric("Std Deviation", f"{std_concern:.2f}%")
            
            with col_r3:
                st.metric("95% CI", f"[{ci_lower:.1f}%, {ci_upper:.1f}%]")
            
            # Interpretation
            if std_concern < 2:
                robustness = "Very Robust"
                rob_color = "#2e7d32"
            elif std_concern < 5:
                robustness = "Robust"
                rob_color = "#689f38"
            elif std_concern < 10:
                robustness = "Moderate"
                rob_color = "#f9a825"
            else:
                robustness = "Sensitive"
                rob_color = "#ef6c00"
            
            st.markdown(
                f"""
                <div style='padding:0.8rem;border-radius:6px;background:{rob_color}20;border:1px solid {rob_color};margin:1rem 0;'>
                    <strong>Robustness:</strong> {robustness}
                    <div style='opacity:0.8;font-size:0.9rem;margin-top:0.3rem;'>
                    A low standard deviation indicates the score is stable despite measurement errors.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Histogram
            st.markdown("**Distribution of Concern Scores:**")
            hist_data = pd.DataFrame({
                "Concern (%)": mc_results
            })
            
            chart = alt.Chart(hist_data).mark_bar(opacity=0.7).encode(
                alt.X("Concern (%):Q", bin=alt.Bin(maxbins=30), title="Concern (%)"),
                alt.Y("count()", title="Frequency"),
                tooltip=["count()"]
            ).properties(height=300)
            
            # Add vertical line for base score
            base_line = alt.Chart(pd.DataFrame({"x": [base_concern_mc]})).mark_rule(
                color="red", strokeDash=[5, 5], size=2
            ).encode(x="x:Q")
            
            st.altair_chart(chart + base_line, use_container_width=True)
            
            # Sensitivity breakdown
            st.markdown("**Sensitivity Analysis:**")
            st.markdown("Which inputs contribute most to output uncertainty?")
            
            # Compute variance contribution for each input
            sensitivities = []
            saved_inputs = stats["inputs"]
            saved_uncertainties = stats["uncertainties"]
            for var_name in saved_inputs.keys():
                # Run mini MC varying only this input
                mini_results = []
                for _ in range(100):
                    test_inputs = saved_inputs.copy()
                    mf = registry.get(var_name)
                    vmin, vmax = float(mf.values[0]), float(mf.values[-1])
                    unc = saved_uncertainties.get(var_name, 0.0)
                    perturbed = np.random.normal(saved_inputs[var_name], unc)
                    test_inputs[var_name] = float(np.clip(perturbed, vmin, vmax))
                    mini_result = engine.evaluate(test_inputs)
                    mini_results.append(mini_result["concern_percent"])
                
                var_contribution = np.std(mini_results)
                sensitivities.append({
                    "Variable": var_name,
                    "Uncertainty (±)": f"{saved_uncertainties.get(var_name, 0):.2f}",
                    "Output Variance Contribution (%)": f"{var_contribution:.2f}"
                })
            
            sens_df = pd.DataFrame(sensitivities).sort_values(
                "Output Variance Contribution (%)", 
                ascending=False,
                key=lambda x: x.str.replace("%", "").astype(float)
            )
            st.dataframe(sens_df, use_container_width=True, hide_index=True)
            
            st.info("**Interpretation:** Variables with higher variance contribution have more impact on uncertainty. Consider measuring these more carefully.")

st.caption("Notes: values outside the supported range are clipped; the percentage simply rescales the output to 0–100.")
