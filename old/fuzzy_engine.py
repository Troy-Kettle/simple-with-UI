import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd


@dataclass
class MembershipFunctionSet:
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
        values = df["Value"].to_numpy(dtype=float)
        matrix = df[labels].to_numpy(dtype=float)
        return cls(name=name, values=values, labels=labels, matrix=matrix)

    def membership(self, label: str, x: float) -> float:
        # Linear interpolation on the provided grid (no parametric simplification)
        if label not in self.labels:
            raise KeyError(f"Label '{label}' not found for variable '{self.name}'")
        idx = self.labels.index(label)
        xv = self.values
        yv = self.matrix[:, idx]
        # clamp outside range
        if x <= xv[0]:
            return float(yv[0])
        if x >= xv[-1]:
            return float(yv[-1])
        # find interval
        i = np.searchsorted(xv, x) - 1
        x0, x1 = xv[i], xv[i+1]
        y0, y1 = yv[i], yv[i+1]
        if x1 == x0:
            return float(y0)
        t = (x - x0) / (x1 - x0)
        return float((1.0 - t) * y0 + t * y1)

    def all_memberships(self, x: float) -> Dict[str, float]:
        return {label: self.membership(label, x) for label in self.labels}


class FuzzyVariableRegistry:
    def __init__(self, variables: Dict[str, MembershipFunctionSet]):
        self._vars = variables

    @classmethod
    def from_directory(cls, csv_dir: str) -> "FuzzyVariableRegistry":
        variables: Dict[str, MembershipFunctionSet] = {}
        for fname in os.listdir(csv_dir):
            if not fname.endswith(".csv"):
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
            var_name = var_name.title()
            mfset = MembershipFunctionSet.from_csv(var_name, path)
            variables[var_name] = mfset
        if not variables:
            raise ValueError(f"No membership function CSVs found in {csv_dir}")
        return cls(variables)

    def get(self, name: str) -> MembershipFunctionSet:
        # Accept case-insensitive and basic normalization
        key = self._normalize(name)
        for k in self._vars.keys():
            if self._normalize(k) == key:
                return self._vars[k]
        raise KeyError(f"Variable '{name}' not found. Available: {list(self._vars.keys())}")

    def variable_names(self) -> List[str]:
        return list(self._vars.keys())

    @staticmethod
    def _normalize(s: str) -> str:
        return " ".join(s.strip().lower().split())


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
        values = df["Value"].to_numpy(dtype=float)
        matrix = df[labels].to_numpy(dtype=float)
        return cls(values=values, labels=labels, matrix=matrix)

    def vector_for_label(self, label: str) -> np.ndarray:
        if label not in self.labels:
            raise KeyError(f"Output label '{label}' not found. Available: {self.labels}")
        idx = self.labels.index(label)
        return self.matrix[:, idx]


@dataclass
class Rule:
    rule_id: str
    logic: str  # AND or OR
    antecedent: List[Tuple[str, str]]  # list of (variable, label)
    consequent_label: str
    weight: float = 1.0  # optional scaling of firing strength (0..1+ allowed but clipped later)

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


class MamdaniEngine:
    def __init__(self, registry: FuzzyVariableRegistry, output_mf: ConcernOutputMF, rules: RuleBase, aggregation: str = "max"):
        self.registry = registry
        self.output = output_mf
        self.rules = rules
        agg = aggregation.strip().lower()
        if agg not in {"max", "prob_sum"}:
            agg = "max"
        self.aggregation = agg

    def evaluate(self, crisp_inputs: Dict[str, float]) -> Dict:
        # 1) fuzzify inputs
        input_memberships: Dict[str, Dict[str, float]] = {}
        for var_name, x in crisp_inputs.items():
            mfset = self.registry.get(var_name)
            input_memberships[var_name] = mfset.all_memberships(float(x))

        # 2) rule evaluation (min for AND, max for OR)
        firing_info: List[Dict] = []
        # Collect per-label clipped vectors to aggregate later
        label_clipped_vectors: Dict[str, List[np.ndarray]] = {l: [] for l in self.output.labels}

        for rule in self.rules.rules:
            degrees: List[float] = []
            antecedent_details: List[Dict] = []
            for (var, label) in rule.antecedent:
                mfset = self.registry.get(var)
                x = crisp_inputs[var if var in crisp_inputs else mfset.name]
                deg = mfset.membership(label, float(x))
                degrees.append(deg)
                antecedent_details.append({"variable": mfset.name, "label": label, "degree": deg})

            if not degrees:
                # Empty antecedent means 'always true' baseline rule
                fire = 1.0
            elif rule.logic == "AND":
                fire = float(np.min(degrees))
            else:
                fire = float(np.max(degrees))

            # Apply optional rule weight, then clip to [0,1]
            fire = float(np.clip(fire * max(0.0, rule.weight), 0.0, 1.0))

            # collect firing
            firing_info.append({
                "RuleID": rule.rule_id,
                "Logic": rule.logic,
                "Antecedents": antecedent_details,
                "ConsequentLabel": rule.consequent_label,
                "FiringStrength": fire,
            })

            # 3) implication (min) and 4) aggregation performed later; collect clipped vectors per label
            if fire > 0.0:
                base = self.output.vector_for_label(rule.consequent_label)
                clipped_vec = np.minimum(base, fire)
                label_clipped_vectors[rule.consequent_label].append(clipped_vec)

        # Aggregate output by taking pointwise max over all clipped consequents
        agg = np.zeros_like(self.output.values, dtype=float)
        per_label_agg: Dict[str, np.ndarray] = {}
        for label in self.output.labels:
            vecs = label_clipped_vectors[label]
            if not vecs:
                combined = np.zeros_like(self.output.values, dtype=float)
            else:
                if self.aggregation == "max":
                    combined = np.maximum.reduce(vecs)
                else:  # probabilistic sum: 1 - Π(1 - v)
                    one_minus = [1.0 - np.clip(v, 0.0, 1.0) for v in vecs]
                    prod = np.ones_like(one_minus[0])
                    for om in one_minus:
                        prod *= om
                    combined = 1.0 - prod
            per_label_agg[label] = combined
            agg = np.maximum(agg, combined)

        # 5) defuzzify via centroid on provided Value universe
        num = float(np.trapz(agg * self.output.values, self.output.values))
        den = float(np.trapz(agg, self.output.values))
        if den > 0:
            centroid = num / den
        else:
            centroid = 0.0

        return {
            "concern_value": centroid,
            "input_memberships": input_memberships,
            "rule_firings": firing_info,
            "aggregated_output": {k: v.tolist() for k, v in per_label_agg.items()},
        }
