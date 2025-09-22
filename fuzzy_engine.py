import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Membership Functions
# -----------------------------

@dataclass
class MembershipFunctionSet:
    """
    A set of fuzzy membership functions defined on a shared 1D universe of discourse.
    - `values`: 1D grid (monotonic increasing)
    - `labels`: list of linguistic labels (columns in CSV other than "Value")
    - `matrix`: shape (N, L) membership values in [0,1]
    """
    name: str
    values: np.ndarray  # shape (N,)
    labels: List[str]   # list of column names (labels)
    matrix: np.ndarray  # shape (N, L) membership values per label

    @classmethod
    def from_csv(cls, name: str, path: str) -> "MembershipFunctionSet":
        df = pd.read_csv(path)
        if "Value" not in df.columns:
            raise ValueError(f"CSV {path} must contain a 'Value' column")
        labels = [c for c in df.columns if c != "Value"]
        if not labels:
            raise ValueError(f"CSV {path} must include at least one label column")
        # Ensure Value is sorted strictly increasing (required for interpolation)
        df = df.sort_values("Value", kind="mergesort").reset_index(drop=True)
        values = df["Value"].to_numpy(dtype=float)
        matrix = df[labels].to_numpy(dtype=float)

        # Clip any small floating error outside [0,1] and warn via attribute later
        matrix = np.clip(matrix, 0.0, 1.0)

        return cls(name=name, values=values, labels=labels, matrix=matrix)

    def _interp_column(self, yv: np.ndarray, x: float, outside: str = "edge") -> float:
        """
        Piecewise-linear interpolation with outside-domain handling.
        outside: "edge" (return boundary value) or "zero" (return 0 outside).
        """
        xv = self.values
        # Outside handling
        if x <= xv[0]:
            return 0.0 if outside == "zero" else float(yv[0])
        if x >= xv[-1]:
            return 0.0 if outside == "zero" else float(yv[-1])

        # Find interval
        i = np.searchsorted(xv, x) - 1
        if i < 0:
            i = 0
        if i >= len(xv) - 1:
            i = len(xv) - 2
        x0, x1 = xv[i], xv[i+1]
        y0, y1 = yv[i], yv[i+1]
        if x1 == x0:
            return float(y0)
        t = (x - x0) / (x1 - x0)
        return float((1.0 - t) * y0 + t * y1)

    def membership(self, label: str, x: float, outside: str = "edge") -> float:
        """Return μ_label(x) with selected outside-domain policy."""
        if label not in self.labels:
            raise KeyError(f"Label '{label}' not found for variable '{self.name}'. Available: {self.labels}")
        idx = self.labels.index(label)
        yv = self.matrix[:, idx]
        return self._interp_column(yv, float(x), outside=outside)

    def all_memberships(self, x: float, outside: str = "edge") -> Dict[str, float]:
        """Return a dict {label: μ_label(x)} for all labels in this set."""
        return {label: self.membership(label, x, outside=outside) for label in self.labels}

    # --- Data quality helpers ---
    def coverage_stats(self) -> Dict[str, float]:
        """
        Compute simple diagnostics useful for sanity checking CSV MF quality.
        - avg_max: mean over x of max_label μ(x)  (should be close to 1.0)
        - min_max: minimum over x of max_label μ(x)
        - max_gap: maximum gap 1 - max_label μ(x)
        """
        max_over_labels = np.max(self.matrix, axis=1)
        avg_max = float(np.mean(max_over_labels))
        min_max = float(np.min(max_over_labels))
        max_gap = float(np.max(1.0 - max_over_labels))
        return {"avg_max": avg_max, "min_max": min_max, "max_gap": max_gap}

    def domain(self) -> Tuple[float, float]:
        return float(self.values[0]), float(self.values[-1])


# -----------------------------
# Registry for variables
# -----------------------------

class FuzzyVariableRegistry:
    def __init__(self, variables: Dict[str, MembershipFunctionSet]):
        self._vars = variables

    @classmethod
    def from_directory(cls, csv_dir: str) -> "FuzzyVariableRegistry":
        variables: Dict[str, MembershipFunctionSet] = {}
        for fname in os.listdir(csv_dir):
            if not fname.lower().endswith(".csv"):
                continue
            lower = fname.lower()
            if lower in {"rules.csv", "concern_output_membership_functions.csv"}:
                continue
            if not lower.endswith("_membership_functions.csv"):
                # skip unrelated CSVs
                continue
            path = os.path.join(csv_dir, fname)
            var_name = lower.replace("_membership_functions.csv", "")
            var_name = var_name.replace("_", " ").strip()
            var_name = titlecase(var_name)
            mfset = MembershipFunctionSet.from_csv(var_name, path)
            variables[var_name] = mfset
        if not variables:
            raise ValueError(f"No membership function CSVs found in {csv_dir}")
        return cls(variables)

    def get(self, name: str) -> MembershipFunctionSet:
        key = _normalize(name)
        for k in self._vars.keys():
            if _normalize(k) == key:
                return self._vars[k]
        # Allow simple synonyms for common vitals
        alias = COMMON_ALIASES.get(key)
        if alias:
            return self.get(alias)
        raise KeyError(f"Variable '{name}' not found. Available: {list(self._vars.keys())}")

    def variable_names(self) -> List[str]:
        return list(self._vars.keys())

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {k: v.coverage_stats() for k, v in self._vars.items()}


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def titlecase(s: str) -> str:
    return " ".join([w.capitalize() for w in s.split()])


COMMON_ALIASES = {
    _normalize("SpO2"): _normalize("oxygen saturation"),
    _normalize("SaO2"): _normalize("oxygen saturation"),
    _normalize("O2 Sat"): _normalize("oxygen saturation"),
    _normalize("Heart Rate (HR)"): _normalize("heart rate"),
    _normalize("RR"): _normalize("respiratory rate"),
    _normalize("SBP"): _normalize("systolic blood pressure"),
    _normalize("FiO2"): _normalize("inspired oxygen concentration"),
}


# -----------------------------
# Output MFs
# -----------------------------

@dataclass
class ConcernOutputMF:
    values: np.ndarray  # shape (N,)
    labels: List[str]
    matrix: np.ndarray  # shape (N, L)

    @classmethod
    def from_csv(cls, path: str) -> "ConcernOutputMF":
        df = pd.read_csv(path)
        if "Value" not in df.columns:
            raise ValueError("Concern output CSV must contain a 'Value' column")
        labels = [c for c in df.columns if c != "Value"]
        if not labels:
            raise ValueError("Concern output CSV must include at least one label column")
        df = df.sort_values("Value", kind="mergesort").reset_index(drop=True)
        values = df["Value"].to_numpy(dtype=float)
        matrix = np.clip(df[labels].to_numpy(dtype=float), 0.0, 1.0)
        return cls(values=values, labels=labels, matrix=matrix)

    def vector_for_label(self, label: str) -> np.ndarray:
        if label not in self.labels:
            raise KeyError(f"Output label '{label}' not found. Available: {self.labels}")
        idx = self.labels.index(label)
        return self.matrix[:, idx]

    def scale_to_percent(self, x: float) -> float:
        vmin, vmax = float(self.values[0]), float(self.values[-1])
        if vmax == vmin:
            return 0.0
        # Map to [0,100] for display only
        return 100.0 * (x - vmin) / (vmax - vmin)


# -----------------------------
# Rules
# -----------------------------

@dataclass
class Rule:
    rule_id: str
    logic: str  # AND or OR
    antecedent: List[Tuple[str, str]]  # list of (variable, label)
    consequent_label: str
    weight: float = 1.0  # optional scaling of firing strength

    def __post_init__(self):
        lg = self.logic.strip().upper()
        if lg not in {"AND", "OR"}:
            raise ValueError("Rule.logic must be 'AND' or 'OR'")
        self.logic = lg


class RuleBase:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    @classmethod
    def from_csv(cls, path: str) -> "RuleBase":
        df = pd.read_csv(path)
        required = {"RuleID", "Logic", "Antecedent", "ConsequentLabel"}
        if not required.issubset(df.columns):
            raise ValueError(f"Rules CSV must have columns: {sorted(required)}")
        rules: List[Rule] = []
        for _, row in df.iterrows():
            rid = str(row["RuleID"]) if not pd.isna(row["RuleID"]) else ""
            logic = str(row["Logic"]).strip()
            antecedent_json = row["Antecedent"]
            try:
                antecedent_list = json.loads(antecedent_json)
                antecedent: List[Tuple[str, str]] = []
                for obj in antecedent_list:
                    var = obj["variable"]
                    label = obj["label"]
                    antecedent.append((var, label))
            except Exception as e:
                raise ValueError(f"Invalid Antecedent JSON for RuleID {rid}: {e}")
            cons = str(row["ConsequentLabel"]).strip()
            weight = 1.0
            if "Weight" in df.columns and not pd.isna(row.get("Weight", np.nan)):
                try:
                    weight = float(row.get("Weight", 1.0))
                except Exception:
                    weight = 1.0
            rules.append(Rule(rule_id=rid, logic=logic, antecedent=antecedent, consequent_label=cons, weight=weight))
        return cls(rules)

    # --- Audits ---
    def audit(self, registry: "FuzzyVariableRegistry", output: "ConcernOutputMF") -> Dict[str, List[str]]:
        """Return dict with lists of human-readable issues found in rules."""
        issues = {"unknown_variables": [], "unknown_labels": [], "unknown_output_labels": [], "empty_antecedents": []}
        known_vars = { _normalize(n): n for n in registry.variable_names() }
        for r in self.rules:
            if not r.antecedent:
                issues["empty_antecedents"].append(r.rule_id or "<no id>")
            for (v, lab) in r.antecedent:
                vn = _normalize(v)
                # resolve alias
                if vn in COMMON_ALIASES:
                    vn = COMMON_ALIASES[vn]
                if vn not in known_vars:
                    issues["unknown_variables"].append(f"Rule {r.rule_id}: variable '{v}' not found")
                else:
                    # label check
                    mf = registry.get(known_vars[vn])
                    if lab not in mf.labels:
                        issues["unknown_labels"].append(f"Rule {r.rule_id}: label '{lab}' not in '{mf.name}' ({mf.labels})")
            if r.consequent_label not in output.labels:
                issues["unknown_output_labels"].append(f"Rule {r.rule_id}: consequent '{r.consequent_label}' not in output labels {output.labels}")
        # Deduplicate
        for k in issues:
            issues[k] = sorted(list(dict.fromkeys(issues[k])))
        return issues


# -----------------------------
# Mamdani Engine
# -----------------------------

class MamdaniEngine:
    def __init__(
        self,
        registry: FuzzyVariableRegistry,
        output_mf: ConcernOutputMF,
        rules: RuleBase,
        aggregation: str = "max",
        outside_behavior: str = "zero",   # safer clinical default
    ):
        self.registry = registry
        self.output = output_mf
        self.rules = rules
        agg = aggregation.strip().lower()
        if agg not in {"max", "prob_sum"}:
            agg = "max"
        self.aggregation = agg
        if outside_behavior not in {"edge", "zero"}:
            outside_behavior = "zero"
        self.outside_behavior = outside_behavior

    def _fuzzify_all(self, crisp_inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for var_name, x in crisp_inputs.items():
            mfset = self.registry.get(var_name)
            out[var_name] = mfset.all_memberships(float(x), outside=self.outside_behavior)
        return out

    def _defuzzify_centroid(self, agg_vec: np.ndarray) -> float:
        num = float(np.trapz(agg_vec * self.output.values, self.output.values))
        den = float(np.trapz(agg_vec, self.output.values))
        return num / den if den > 0 else 0.0

    def evaluate(self, crisp_inputs: Dict[str, float], include_sensitivity: bool = True) -> Dict:
        # 1) Fuzzify
        input_memberships: Dict[str, Dict[str, float]] = self._fuzzify_all(crisp_inputs)

        # 2) Rule evaluation (min for AND, max for OR)
        firing_info: List[Dict] = []
        label_clipped_vectors: Dict[str, List[np.ndarray]] = {l: [] for l in self.output.labels}

        for rule in self.rules.rules:
            degrees: List[float] = []
            antecedent_details: List[Dict] = []
            for (var, label) in rule.antecedent:
                mfset = self.registry.get(var)
                # resolve input key to registry standardized name
                x = crisp_inputs.get(mfset.name, None)
                if x is None:
                    # final fallback: try provided var literal
                    x = float(crisp_inputs.get(var, 0.0))
                deg = mfset.membership(label, float(x), outside=self.outside_behavior)
                degrees.append(deg)
                antecedent_details.append({"variable": mfset.name, "label": label, "degree": deg})

            if not degrees:
                fire = 1.0
            elif rule.logic == "AND":
                fire = float(np.min(degrees))
            else:
                fire = float(np.max(degrees))

            fire = float(np.clip(fire * max(0.0, rule.weight), 0.0, 1.0))

            firing_info.append({
                "RuleID": rule.rule_id,
                "Logic": rule.logic,
                "Antecedents": antecedent_details,
                "ConsequentLabel": rule.consequent_label,
                "FiringStrength": fire,
            })

            if fire > 0.0:
                base = self.output.vector_for_label(rule.consequent_label)
                clipped_vec = np.minimum(base, fire)
                label_clipped_vectors[rule.consequent_label].append(clipped_vec)

        # 3) Aggregate per label, then global aggregate
        agg = np.zeros_like(self.output.values, dtype=float)
        per_label_agg: Dict[str, np.ndarray] = {}
        for label in self.output.labels:
            vecs = label_clipped_vectors[label]
            if not vecs:
                combined = np.zeros_like(self.output.values, dtype=float)
            else:
                if self.aggregation == "max":
                    combined = np.maximum.reduce(vecs)
                else:
                    one_minus = [1.0 - np.clip(v, 0.0, 1.0) for v in vecs]
                    prod = np.ones_like(one_minus[0])
                    for om in one_minus:
                        prod *= om
                    combined = 1.0 - prod
            per_label_agg[label] = combined
            agg = np.maximum(agg, combined)

        # 4) Defuzzify
        centroid = self._defuzzify_centroid(agg)
        centroid_pct = self.output.scale_to_percent(centroid)

        result = {
            "concern_value": centroid,                 # in native output units
            "concern_percent": centroid_pct,           # 0-100 scaled for display
            "input_memberships": input_memberships,
            "rule_firings": firing_info,
            "aggregated_output": {k: v.tolist() for k, v in per_label_agg.items()},
            "output_domain": (float(self.output.values[0]), float(self.output.values[-1])),
        }

        # Optional: local sensitivities (small nudges up/down for each input)
        if include_sensitivity:
            sens = {}
            for var_name, x in crisp_inputs.items():
                mfset = self.registry.get(var_name)
                lo, hi = mfset.domain()
                span = hi - lo
                if span <= 0:
                    sens[var_name] = {"delta_up": 0.0, "delta_down": 0.0}
                    continue
                step = 0.02 * span  # 2% of domain
                def clamp(v): 
                    return float(np.clip(v, lo, hi))
                # up
                inputs_up = dict(crisp_inputs)
                inputs_up[mfset.name] = clamp(x + step)
                up_val = self.evaluate(inputs_up, include_sensitivity=False)["concern_value"]
                # down
                inputs_dn = dict(crisp_inputs)
                inputs_dn[mfset.name] = clamp(x - step)
                dn_val = self.evaluate(inputs_dn, include_sensitivity=False)["concern_value"]
                sens[var_name] = {"delta_up": float(up_val - centroid), "delta_down": float(centroid - dn_val)}
            result["local_sensitivity"] = sens

        return result


# -----------------------------
# Clinical QA helpers (lightweight)
# -----------------------------

EXPECTED_DIRECTIONS = {
    # meaning: how Concern should trend as the variable increases (holding others near normal)
    # "increasing": higher value => higher concern
    # "decreasing": higher value => lower concern
    # "u_shaped": risk at both extremes (no monotonic expectation)
    # Unlisted variables default to "unknown" (no check).
    "inspired oxygen concentration": "increasing",
    "supplementary oxygen": "increasing",
    "oxygen saturation": "decreasing",
    "respiratory rate": "increasing",
    "systolic blood pressure": "decreasing",
    # heart rate, temperature often u-shaped clinically
    "heart rate": "u_shaped",
    "temperature": "u_shaped",
}

def sweep_monotonicity(engine: MamdaniEngine, var_name: str, baseline: Dict[str, float], n: int = 25) -> Dict[str, float]:
    """Sweep one variable from min..max while keeping others fixed; return slope and simple quality stats."""
    mf = engine.registry.get(var_name)
    xs = np.linspace(mf.values[0], mf.values[-1], n)
    ys = []
    for x in xs:
        inp = dict(baseline)
        inp[mf.name] = float(x)
        ys.append(engine.evaluate(inp, include_sensitivity=False)["concern_value"])
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # Simple linear trend (least squares slope)
    slope = float(np.polyfit(xs, ys, 1)[0]) if n >= 2 else 0.0
    # Spearman-like sign using ranks without SciPy: compute correlation of ranks
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    rho = float((rx * ry).sum() / denom) if denom > 0 else 0.0
    return {"slope": slope, "rho": rho, "x_min": float(xs[0]), "x_max": float(xs[-1])}

def direction_check(var_key: str, rho: float) -> Optional[bool]:
    """Return True/False for pass/fail if expectation exists; None if unknown/u-shaped (skip)."""
    exp = EXPECTED_DIRECTIONS.get(_normalize(var_key))
    if exp == "increasing":
        return rho >= 0.3   # allow some noise
    if exp == "decreasing":
        return rho <= -0.3
    return None
