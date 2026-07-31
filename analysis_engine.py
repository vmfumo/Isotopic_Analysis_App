import os
import re
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem

# Core runtime bootstrap loader to ensure molmass is present
try:
    from molmass import Formula
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "molmass"])
    from molmass import Formula

HAS_DRAW = False
try:
    from rdkit.Chem import Draw
    HAS_DRAW = True
except ImportError:
    HAS_DRAW = False

def parse_formula(formula_str):
    """Parses a chemical formula string into an elemental dictionary."""
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    element_dict = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_dict[element] = element_dict.get(element, 0) + count
    return element_dict

def get_natural_distribution(smiles, substituent_smiles=None):
    """Calculates the baseline natural abundance isotope profile for a compound structure."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return pd.DataFrame(columns=["m/z", "Normalized Abundance (%)"])
        
    if substituent_smiles:
        sub = Chem.MolFromSmiles(substituent_smiles)
        if not sub:
            return pd.DataFrame(columns=["m/z", "Normalized Abundance (%)"])
        combined = Chem.CombineMols(mol, sub)
        mol = Chem.DeleteSubstructs(combined, Chem.MolFromSmiles("[H]"))
        
    formula_str = rdMolDescriptors.CalcMolFormula(mol)
    f = Formula(formula_str)
    
    mz_list = []
    abundance_list = []
    
    # Try modern .isotopes property, fallback to .isotope_distribution() if legacy
    try:
        distribution = f.isotopes
    except AttributeError:
        distribution = f.isotope_distribution()
    
    for mass, abundance in distribution:
        mz_list.append(round(mass))
        abundance_list.append(abundance * 100.0)
        
    df = pd.DataFrame({"m/z": mz_list, "Abundance": abundance_list})
    df = df.groupby("m/z", as_index=False).sum()
    
    if not df.empty and df["Abundance"].max() > 0:
        df["Normalized Abundance (%)"] = (df["Abundance"] / df["Abundance"].max()) * 100.0
    else:
        df["Normalized Abundance (%)"] = 0.0
        
    return df[["m/z", "Normalized Abundance (%)"]]

def simulate_ms_distribution(smiles, labels, substituent_smiles=None):
    """Computes theoretical MS envelopes by combining natural abundances with binomial probabilities."""
    df_base = get_natural_distribution(smiles, substituent_smiles)
    if df_base.empty:
        return pd.DataFrame(columns=["m/z", "Normalized Abundance (%)"])
        
    if not labels:
        return df_base

    current_dist = {0: 1.0}
    for n, p in labels:
        new_dist = {}
        for shift, prob in current_dist.items():
            for k in range(n + 1):
                bin_prob = (math.comb(n, k)) * (p ** k) * ((1 - p) ** (n - k))
                total_shift = shift + k
                new_dist[total_shift] = new_dist.get(total_shift, 0.0) + (prob * bin_prob)
        current_dist = new_dist

    final_envelope = {}
    for base_mz, base_abundance in zip(df_base["m/z"], df_base["Normalized Abundance (%)"]):
        for shift, mass_prob in current_dist.items():
            target_mz = base_mz + shift
            final_envelope[target_mz] = final_envelope.get(target_mz, 0.0) + (base_abundance * mass_prob)

    mz_final = sorted(list(final_envelope.keys()))
    abundance_final = [final_envelope[m] for m in mz_final]
    
    df_out = pd.DataFrame({"m/z": mz_final, "Abundance": abundance_final})
    if not df_out.empty and df_out["Abundance"].max() > 0:
        df_out["Normalized Abundance (%)"] = (df_out["Abundance"] / df_out["Abundance"].max()) * 100.0
    else:
        df_out["Normalized Abundance (%)"] = 0.0
        
    return df_out[["m/z", "Normalized Abundance (%)"]]