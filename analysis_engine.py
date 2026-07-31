import os
import re
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem

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
    """Calculates the baseline natural abundance isotope profile for a compound structure using pure RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return pd.DataFrame(columns=["m/z", "Normalized Abundance (%)"])
        
    if substituent_smiles:
        sub = Chem.MolFromSmiles(substituent_smiles)
        if not sub:
            return pd.DataFrame(columns=["m/z", "Normalized Abundance (%)"])
        combined = Chem.CombineMols(mol, sub)
        mol = Chem.DeleteSubstructs(combined, Chem.MolFromSmiles("[H]"))
    
    mol = Chem.AddHs(mol)
    
    element_isotopes = {
        "C": {0: 0.9893, 1: 0.0107},
        "H": {0: 0.999885, 1: 0.000115},
        "N": {0: 0.99632, 1: 0.00368},
        "O": {0: 0.99757, 1: 0.00038, 2: 0.00205},
        "F": {0: 1.0},
        "P": {0: 1.0},
        "S": {0: 0.9493, 1: 0.0076, 2: 0.0429, 4: 0.0002},
        "Cl": {0: 0.7578, 2: 0.2422},
        "Br": {0: 0.5069, 2: 0.4931},
        "I": {0: 1.0}
    }
    
    nominal_base_mass = 0
    atom_counts = {}
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        if symbol == "C": nominal_base_mass += 12
        elif symbol == "H": nominal_base_mass += 1
        elif symbol == "N": nominal_base_mass += 14
        elif symbol == "O": nominal_base_mass += 16
        elif symbol == "F": nominal_base_mass += 19
        elif symbol == "P": nominal_base_mass += 31
        elif symbol == "S": nominal_base_mass += 32
        elif symbol == "Cl": nominal_base_mass += 35
        elif symbol == "Br": nominal_base_mass += 79
        elif symbol == "I": nominal_base_mass += 127
        else: nominal_base_mass += round(atom.GetMass())
        
    current_envelope = {0: 1.0}
    
    for symbol, count in atom_counts.items():
        iso_dict = element_isotopes.get(symbol, {0: 1.0})
        for _ in range(count):
            next_envelope = {}
            for shift, prob in current_envelope.items():
                for iso_shift, iso_prob in iso_dict.items():
                    new_shift = shift + iso_shift
                    next_envelope[new_shift] = next_envelope.get(new_shift, 0.0) + (prob * iso_prob)
            current_envelope = next_envelope
            
    mz_list = []
    abundance_list = []
    for shift, prob in current_envelope.items():
        if prob > 0.00001:
            mz_list.append(nominal_base_mass + shift)
            abundance_list.append(prob * 100.0)
            
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