import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls  # Linear optimization package for deconvolution
import analysis_engine as engine  # Hooks into your calculation module

# Configure matplotlib to render cleanly as background canvas surfaces
import matplotlib
matplotlib.use('Agg')

# Guarded extraction of RDKit drawing structures directly inside app context
HAS_DRAW = False
try:
    from rdkit.Chem import Draw
    HAS_DRAW = True
except ImportError:
    HAS_DRAW = False
    # Mock container to prevent NameError exceptions when Draw is called headlessly
    class HeadlessDrawMock:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    Draw = HeadlessDrawMock()

# 1. Page Configuration Setup
st.set_page_config(page_title="MS Isotopomer Simulator", layout="wide")

# 2. Main Canvas Banner Headers
st.title("Isotopic Distribution Simulator")
st.markdown("Predicts MS isotopic profiles of deuterated compounds and their regioisomeric products.")

# 3. Sidebar Input Control Panel
with st.sidebar:
    st.header("1. Ionization Mode")
    ion_mode = st.radio(
        "Select Adduct Type",
        options=["[M+H]+ (Positive Mode)", "[M-H]- (Negative Mode)"],
        index=0,
        help="[M+H]+ shifts masses up (+1 from neutral). [M-H]- shifts masses down (-1 from neutral)."
    )
    
    st.divider()
    st.header("2. Deuterated Reactant")
    hetero_smiles = st.text_input("Reactant SMILES", value="O=C(C1=CC=CN=C1)OC")
    
    st.divider()
    st.subheader("Deuterium Incorporation Table")
    st.caption("Input position, % deuterium incorporation, max potential deuteriums, and whether that position is expected to be unreactive. Up to 10 rows max.")
    
    default_deuterium_data = pd.DataFrame([
        {"Position": "2", "Percent Deuterium (%)": 20, "Max Deuteriums": 1, "Unreactive": False},
        {"Position": "4", "Percent Deuterium (%)": 40, "Max Deuteriums": 1, "Unreactive": False},
        {"Position": "5", "Percent Deuterium (%)": 60, "Max Deuteriums": 1, "Unreactive": False},
        {"Position": "6", "Percent Deuterium (%)": 80, "Max Deuteriums": 1, "Unreactive": False},
    ])
    
    edited_df = st.data_editor(
        default_deuterium_data,
        num_rows="dynamic",
        key="deuterium_table",
        use_container_width=True
    )
    
    if len(edited_df) > 10:
        st.warning("Maximum limit of 10 rows reached. Extra rows will be ignored.")
        edited_df = edited_df.head(10)
    
    st.divider()
    st.header("3. Radical Input")
    radicals_raw = st.text_area(
        "Radical SMILES List (One per line)", 
        value="[CH2]CC\n[Br]"
    )
    
    st.divider()
    st.header("4. Advanced Controls")
    enable_nnls = st.checkbox(
        "Enable NNLS Deconvolution", 
        value=False,
        help="Check this box to run a forward stepwise non-negative least squares regression to determine regioisomeric composition of a sample."
    )

# Clean radical inputs
radical_list = [line.strip() for line in radicals_raw.split("\n") if line.strip()]

# Parse interactive table arrays into (position, n, p, unreactive) parameters
parsed_labels = []
for _, row in edited_df.iterrows():
    pos = str(row.get("Position", "")).strip()
    if not pos:
        continue
    try:
        p_val = float(row.get("Percent Deuterium (%)", 0.0)) / 100.0
        n_val = int(row.get("Max Deuteriums", 1))
        is_unreactive = bool(row.get("Unreactive", False))
        parsed_labels.append((pos, n_val, p_val, is_unreactive))
    except (ValueError, TypeError):
        continue


# --- DATA MANIPULATION & GRAPHING HELPER FUNCTIONS ---

def adjust_ion_mode_masses(df, mode_selection):
    """Adjusts the engine's m/z outputs if Negative Ion Mode [M-H]- is chosen"""
    if df.empty or 'm/z' not in df.columns:
        return df
    if "[M-H]-" in mode_selection:
        df['m/z'] = df['m/z'] - 2
    return df

def trim_high_mass_tail(df):
    """Drops rows after the most intense mass peak if abundance < 0.5%"""
    if df.empty or 'Normalized Abundance (%)' not in df.columns:
        return df
    df_clean = df.reset_index(drop=True)
    peak_idx = df_clean['Normalized Abundance (%)'].idxmax()
    drop_indices = df_clean.index[
        (df_clean.index > peak_idx) & 
        (df_clean['Normalized Abundance (%)'] < 0.5)
    ]
    if len(drop_indices) > 0:
        df_clean = df_clean.iloc[:drop_indices[0]]
    return df_clean

def generate_locked_ms_plot(df, title_label, intensity_col='Normalized Abundance (%)', color='#1f77b4'):
    """Generates a static bar chart with a locked 0-100 y-axis"""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df['m/z'], df[intensity_col], color=color, width=0.4, edgecolor='black', zorder=3)
    ax.set_title(title_label, fontsize=10, fontweight='bold')
    ax.set_xlabel('m/z', fontsize=9)
    ax.set_ylabel(intensity_col, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xticks(df['m/z'])
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    fig.tight_layout()
    return fig


# --- EXPERIMENTAL DATA BLOCKS SECTION ---
exp_data_dict = {}
if enable_nnls:
    st.header("Experimental Mass Distribution Entry")
    st.markdown("Provide observed experimental MS distributions for each radical below to run the NNLS deconvolution.")

    if radical_list:
        tabs = st.tabs([f"Radical: {rad}" for rad in radical_list])
        for tab, rad_smiles in zip(tabs, radical_list):
            with tab:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Input Data Table (`{rad_smiles}`)**")
                    template_df = pd.DataFrame({"m/z": [180, 181, 182, 183, 184], "Abundance": [23.7, 94.4, 100, 13.8, 1.3]})
                    
                    exp_edited = st.data_editor(
                        template_df,
                        num_rows="dynamic",
                        key=f"exp_table_{rad_smiles}",
                        use_container_width=True
                    )
                    exp_data_dict[rad_smiles] = exp_edited
                with col2:
                    try:
                        rad_mol = engine.Chem.MolFromSmiles(rad_smiles)
                        if rad_mol:
                            st.markdown("**Radical Structure**")
                            st.image(Draw.MolToImage(engine.Chem.RemoveHs(rad_mol), size=(200, 120)), width=150)
                    except:
                        pass
    else:
        st.info("Define your working radical SMILES lists in the sidebar to generate experimental data entry fields.")
    st.divider()

# 4. Calculation and Rendering Logic Execution Panel
button_label = "Run Simulation & Deconvolution" if enable_nnls else "Run Simulation"
if st.sidebar.button(button_label, type="primary"):
    if not hetero_smiles:
        st.error("Please enter a valid Heteroarene SMILES string.")
    elif not parsed_labels:
        st.error("Please provide at least one valid row in the deuterium incorporation table.")
    else:
        try:
            # --- STRUCTURE AND NATURAL ABUNDANCE ---
            hetero_mol = engine.Chem.MolFromSmiles(hetero_smiles)
            if hetero_mol:
                st.subheader("Structure and Natural Abundance Profile")
                img_mol = engine.Chem.RemoveHs(hetero_mol)
                engine.Chem.FindPotentialStereoBonds(img_mol)
                mol_image = Draw.MolToImage(img_mol, size=(600, 400), kekulize=True, wedgeBonds=True)
                
                df_natural = engine.simulate_ms_distribution(hetero_smiles, [])
                if 'Relative Abundance (%)' in df_natural.columns:
                    df_natural = df_natural.drop(columns=['Relative Abundance (%)'])
                df_natural = adjust_ion_mode_masses(df_natural, ion_mode)
                df_natural = trim_high_mass_tail(df_natural)
                
                vis_col1, vis_col2 = st.columns([1, 1])
                with vis_col1:
                    st.image(mol_image, caption="Core Structure", width=320)
                with vis_col2:
                    st.markdown(f"**Natural Isotopic Abundance Data ({ion_mode.split(' ')[0]})**")
                    st.dataframe(df_natural.set_index('m/z'), use_container_width=True)
            else:
                raise ValueError("Could not parse the provided Heteroarene SMILES string.")
            
            st.divider()
            st.header("Simulated Mass Distribution Results")
            
            # --- PHASE A: UNREACTED STARTING MATERIAL ---
            st.subheader("1. Unreacted Starting Material")
            df_sm = engine.simulate_ms_distribution(hetero_smiles, [(n, p) for _, n, p, _ in parsed_labels])
            if 'Relative Abundance (%)' in df_sm.columns:
                df_sm = df_sm.drop(columns=['Relative Abundance (%)'])
            df_sm = adjust_ion_mode_masses(df_sm, ion_mode)
            df_sm = trim_high_mass_tail(df_sm)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(df_sm.set_index('m/z'), use_container_width=True)
            with col2:
                fig_sm = generate_locked_ms_plot(df_sm, f"Starting Material ({ion_mode.split(' ')[0]})")
                st.pyplot(fig_sm)
                plt.close(fig_sm)
                
            st.divider()
            
            # --- PHASE B: REGIOISOMERIC PRODUCT ENVELOPES ---
            st.subheader("2. Product Regioisomers" + (" & NNLS Deconvolution" if enable_nnls else ""))
            
            if not radical_list:
                st.warning("No radicals entered.")
            else:
                for rad_smiles in radical_list:
                    st.divider()
                    
                    # Create a side-by-side header row: Title text on the left, Structure Thumbnail on the right
                    rad_col_title, rad_col_img = st.columns([3, 1])
                    
                    with rad_col_title:
                        st.markdown(f"### Calculated Distributions for Radical: `{rad_smiles}`")
                    
                    with rad_col_img:
                        try:
                            rad_mol = engine.Chem.MolFromSmiles(rad_smiles)
                            if rad_mol:
                                rad_img_mol = engine.Chem.RemoveHs(rad_mol)
                                rad_thumb = Draw.MolToImage(rad_img_mol, size=(200, 100), kekulize=True, wedgeBonds=True)
                                st.image(rad_thumb, use_container_width=False, width=140)
                        except:
                            pass # Fail silently if an atypical fragment cannot be visualized, maintaining layout flow
                    
                    theoretical_profiles = {}
                    
                    for target_pos, n, p, unreactive in parsed_labels:
                        if n >= 1:
                            if unreactive:
                                continue
                                
                            product_labels = [(n_i, p_i) for pos_i, n_i, p_i, _ in parsed_labels if pos_i != target_pos]
                            df_product = engine.simulate_ms_distribution(
                                smiles=hetero_smiles,
                                labels=product_labels,
                                substituent_smiles=rad_smiles
                            )
                            if 'Relative Abundance (%)' in df_product.columns:
                                df_product = df_product.drop(columns=['Relative Abundance (%)'])
                            df_product = adjust_ion_mode_masses(df_product, ion_mode)
                            df_product = trim_high_mass_tail(df_product)
                            
                            theoretical_profiles[target_pos] = df_product
                            
                            with st.expander(f"Position {target_pos} Distribution", expanded=False):
                                p_col1, p_col2 = st.columns([1, 2])
                                with p_col1:
                                    st.dataframe(df_product.set_index('m/z'), use_container_width=True)
                                with p_col2:
                                    fig_prod = generate_locked_ms_plot(df_product, f"Position {target_pos} Regioisomer ({ion_mode.split(' ')[0]})")
                                    st.pyplot(fig_prod)
                                    plt.close(fig_prod)
                    
                    # --- NNLS DECONVOLUTION SOLVER STEP (CONDITIONAL) ---
                    if enable_nnls:
                        st.markdown("#### Forward Stepwise NNLS Regression Results")
                        raw_exp_df = exp_data_dict.get(rad_smiles, pd.DataFrame())
                        
                        if raw_exp_df.empty or 'm/z' not in raw_exp_df.columns or 'Abundance' not in raw_exp_df.columns:
                            st.warning("Missing or unparseable experimental data columns for this radical. Skipping NNLS fitting step.")
                            continue
                            
                        df_exp = raw_exp_df.dropna(subset=['m/z', 'Abundance']).copy()
                        df_exp['m/z'] = df_exp['m/z'].astype(int)
                        df_exp['Abundance'] = df_exp['Abundance'].astype(float)
                        
                        if df_exp['Abundance'].max() > 0:
                            df_exp['Normalized Exp Abundance'] = (df_exp['Abundance'] / df_exp['Abundance'].max()) * 100.0
                        else:
                            df_exp['Normalized Exp Abundance'] = 0.0
                        
                        all_mz_tracks = set(df_exp['m/z'].tolist())
                        for df_t in theoretical_profiles.values():
                            all_mz_tracks.update(df_t['m/z'].astype(int).tolist())
                        sorted_mz = sorted(list(all_mz_tracks))
                        
                        matrix_rows = []
                        regioisomer_order = list(theoretical_profiles.keys())
                        
                        for mz in sorted_mz:
                            row_vals = []
                            for pos in regioisomer_order:
                                df_t = theoretical_profiles[pos]
                                match = df_t[df_t['m/z'].astype(int) == mz]
                                row_vals.append(match['Normalized Abundance (%)'].values[0] if not match.empty else 0.0)
                            matrix_rows.append(row_vals)
                            
                        A = np.array(matrix_rows)
                        
                        b = []
                        for mz in sorted_mz:
                            match = df_exp[df_exp['m/z'] == mz]
                            b.append(match['Normalized Exp Abundance'].values[0] if not match.empty else 0.0)
                        b = np.array(b)
                        
                        if A.size > 0 and len(b) > 0:
                            coefficients, residue = nnls(A, b)
                            total_coeff = np.sum(coefficients)
                            
                            if total_coeff > 0:
                                percentages = (coefficients / total_coeff) * 100.0
                            else:
                                percentages = np.zeros_like(coefficients)
                                
                            results_df = pd.DataFrame({
                                "Regioisomer Position": [f"Position {p}" for p in regioisomer_order],
                                "NNLS Coefficient Fit": coefficients,
                                "Calculated Ratios (%)": percentages
                            })
                            
                            # Apply the filtering logic constraint: Drop entries that contribute less than 1%
                            results_df = results_df[results_df["Calculated Ratios (%)"] >= 1.0]
                            
                            res_col1, res_col2 = st.columns([1, 1])
                            with res_col1:
                                st.markdown("**Regioisomer Ratio Breakdown**")
                                if not results_df.empty:
                                    st.dataframe(results_df.set_index("Regioisomer Position"), use_container_width=True)
                                else:
                                    st.info("No regioisomers met the criteria.")
                                st.metric("Optimization Residual Error Value", f"{residue:.4f}")
                            
                            with res_col2:
                                fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                                
                                if not results_df.empty:
                                    ax_pie.pie(
                                        results_df["Calculated Ratios (%)"], 
                                        labels=results_df["Regioisomer Position"], 
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=plt.cm.tab20.colors,
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
                                    )
                                    ax_pie.axis('equal')
                                else:
                                    ax_pie.text(0.5, 0.5, "No detected regioisomer components", 
                                                ha='center', va='center', fontsize=10, style='italic')
                                    ax_pie.axis('off')
                                    
                                ax_pie.set_title("Calculated Regioisomer Composition", fontsize=11, fontweight='bold')
                                fig_pie.tight_layout()
                                
                                st.pyplot(fig_pie)
                                plt.close(fig_pie)
                        else:
                            st.error("Matrix compilation dimension mismatch error. Check that m/z values match.")
                            
        except Exception as e:
            st.error(f"Execution Halt Error: {str(e)}")
else:
    st.info("Configuration parameters finalized. Click the sidebar execution button to calculate isotope distributions.")