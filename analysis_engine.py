import re
import numpy as np
import pandas as pd
from scipy.stats import binom
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from molmass import Formula

def parse_formula(formula_str):
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    comp = {}
    for elem, count in matches:
        comp[elem] = comp.get(elem, 0) + (int(count) if count else 1)
    return comp

def dict_to_formula(comp):
    return "".join([f"{elem}{comp[elem]}" for elem in sorted(comp.keys()) if comp[elem] > 0])

def parse_label_string_with_positions(label_str):
    """
    Parses a string format like '2:67.6% - 4:3.0% - 5:63.6% - 6:99.1%'
    Returns a list of tuples: [('2', 1, 0.676), ('4', 1, 0.03), ...]
    """
    parsed_labels = []
    if not label_str or pd.isna(label_str): 
        return parsed_labels
    segments = [s.strip() for s in str(label_str).split('-')]
    for segment in segments:
        if not segment: continue
        parts = segment.split(':')
        if len(parts) != 2: continue
        position = parts[0].strip()
        value_str = parts[1].strip()
        match = re.match(r'(?:\((\d+)\))?([\d.]+)%', value_str)
        if not match: continue
        n = int(match.group(1)) if match.group(1) else 1
        p = float(match.group(2)) / 100.0  
        parsed_labels.append((position, n, p))
    return parsed_labels

def simulate_ms_distribution(smiles, labels, substituent_smiles=None):
    """
    Simulates the isotopic profile. 
    To get Starting Material: pass substituent_smiles=None
    To get a Product at a specific position: pass labels for all other positions except the active one.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError("Invalid base Heteroarene SMILES.")
    formula_str = rdMolDescriptors.CalcMolFormula(mol)
    comp = parse_formula(formula_str)

    if substituent_smiles:
        # Subtract one hydrogen for functionalization
        comp['H'] = comp.get('H', 0) - 1
        sub_mol = Chem.MolFromSmiles(substituent_smiles)
        if not sub_mol: raise ValueError(f"Invalid radical SMILES: {substituent_smiles}")
        sub_formula = rdMolDescriptors.CalcMolFormula(sub_mol)
        sub_comp = parse_formula(sub_formula)
        for elem, count in sub_comp.items():
            comp[elem] = comp.get(elem, 0) + count

    base_f = Formula(dict_to_formula(comp))
    spec = base_f.spectrum()
    base_dist = {}
    for peak in spec.values():
        nom_mass = int(round(peak.mass))
        base_dist[nom_mass] = base_dist.get(nom_mass, 0) + peak.fraction

    min_mass = min(base_dist.keys())
    max_mass = max(base_dist.keys())
    dist_array = np.zeros(max_mass - min_mass + 1)
    for m, frac in base_dist.items():
        dist_array[m - min_mass] = frac

    for n, p in labels:
        label_dist = np.array([binom.pmf(k, n, p) for k in range(n + 1)])
        dist_array = np.convolve(dist_array, label_dist)

    results = []
    for i, frac in enumerate(dist_array):
        if frac > 1e-5:  
            results.append({"m/z": min_mass + i + 1, "Relative Abundance (%)": frac * 100})

    df = pd.DataFrame(results)
    if not df.empty:
        df['Normalized Abundance (%)'] = (df['Relative Abundance (%)'] / df['Relative Abundance (%)'].max()) * 100
    return df