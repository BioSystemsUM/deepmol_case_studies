# Using DeepMol models

Models available so far: 

| Model Name                                   | How to Call                     | Prediction Type                                                |
|---------------------------------------------|---------------------------------|----------------------------------------------------------------|
| BBB (Blood-Brain Barrier)                   | `BBB`                  | Penetrates BBB (1) or does not penetrate BBB (0)              |
| AMES Mutagenicity                           | `AMES`                         | Mutagenic (1) or not mutagenic (0)                            |
| Human plasma protein binding rate (PPBR)    | `PPBR`                      | Rate of PPBR expressed in percentage                          |
| Volume of Distribution (VD) at steady state | `VDss`                | Volume of Distribution expressed in liters per kilogram (L/kg)|
| Caco-2 (Cell Effective Permeability)        | `Caco2`                   | Cell Effective Permeability (cm/s)                            |
| HIA (Human Intestinal Absorption)           | `HIA`                      | Absorbed (1) or not absorbed (0)                              |
| Bioavailability                             | `Bioavailability`           | Bioavailable (1) or not bioavailable (0)                      |
| Lipophilicity                               | `Lipophilicity`    | Lipophilicity log-ratio                                       |
| Solubility                                  | `Solubility`           | Solubility (log mol/L)                                        |
| CYP P450 2C9 Inhibition                     | `CYP2C9Inhibition`                 | Inhibit (1) or does not inhibit (0)                           |
| CYP P450 3A4 Inhibition                     | `CYP3A4Inhibition`                 | Inhibit (1) or does not inhibit (0)                           |
| CYP2C9 Substrate                            | `CYP2C9Substrate`| Metabolized (1) or does not metabolize (0)                    |
| CYP2D6 Substrate                            | `CYP2D6Substrate`| Metabolized (1) or does not metabolize (0)                    |
| CYP3A4 Substrate                            | `CYP3A4Substrate`| Metabolized (1) or does not metabolize (0)                    |
| Hepatocyte Clearance                        | `HepatocyteClearance`      | Drug hepatocyte clearance (uL.min-1.(10^6 cells)-1)           |
| NPClassifier                        | `NPClassifier`      | Pathway, Superclass, Class           |
| Plants secondary metabolite precursors predictor                        | `PlantsSMPrecursorPredictor`      | Precursor 1; Precursor 2           |
| Microsome Clearance                 | `MicrosomeClearance`       | Drug microsome clearance (mL.min-1.g-1)          |
| LD50                                | `LD50`        | LD50 (log(1/(mol/kg)))                      |
| hERG Blockers                       | `hERGBlockers`           | hERG blocker (1) or not blocker (0)               |

## How to use:

You can use them either individually or mixed together. 

You can call one model individually, pass a CSV file and get the results in one dataframe:

```python
from deepmol_models import BBB
results = BBB().predict_from_csv("dataset.csv", smiles_field="Drug", id_field="Drug_ID", output_file="predictions.csv")
results
```

| ID | SMILES                                           | BBB Penetration |
|----|-------------------------------------------------|-----------------|
| 0  | OCC(S)CS                                        | 1.0             |
| 1  | CC[N+](C)(C)c1cccc(O)c1                         | 0.0             |
| 2  | Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O | 1.0             |
| 3  | CC(=O)OCC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)CC#N)[...| 0.0             |
| 4  | CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](C(=O)O)c3ccsc3...| 0.0             |


Or pass SMILES strings and get the results in one dataframe:

```python
from deepmol_models import BBB
results = BBB().predict_from_csv("dataset.csv", smiles_field="Drug", id_field="Drug_ID", output_file="predictions.csv")
results
```

| ID | SMILES                                           | BBB Penetration |
|----|-------------------------------------------------|-----------------|
| 1  | CC[N+](C)(C)c1cccc(O)c1                         | 0.0             |
| 2  | Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O | 1.0             |

Complementarily, you can run several models:

```python
from deepmol_models import BBB, PPBR, VDss, Caco2, HIA, Bioavailability, \
    Lipophilicity, Solubility, PlantsSMPrecursorPredictor, NPClassifier, MixedPredictor

# results = MixedPredictor([BBB(), Caco2(), CYP2D6Inhibition(), NPClassifier()]).predict_from_csv("test_molecules.csv", "Drug", "Drug_ID", output_file="predictions.csv")
results = MixedPredictor([BBB(), PPBR(), VDss(), Caco2(), 
                          HIA(), Bioavailability(), Lipophilicity(),
                          Solubility(), PlantsSMPrecursorPredictor(), NPClassifier()]).predict_from_csv("test_molecules.csv", smiles_field="Drug", id_field="Drug_ID", output_file="predictions.csv")
results
```

| ID | SMILES                                           | BBB Penetration | Human PPBR | VDss     | Cell Effective Permeability | Human Intestinal Absorption | Bioavailability | Lipophilicity | Solubility | Precursors                             | Pathways                 | Superclass            | Class                 |
|----|-------------------------------------------------|-----------------|------------|----------|-----------------------------|-----------------------------|-----------------|---------------|------------|---------------------------------------|--------------------------|-----------------------|-----------------------|
| 0  | OCC(S)CS                                        | 1.0             | 64.832665  | 5.803529 | -4.725497                  | 0.0                         | 0.0             | 0.290602      | 0.117866   |                                   | Fatty acyls              | Fatty alcohols       | Fatty alcohols        |
| 1  | CC[N+](C)(C)c1cccc(O)c1                         | 0.0             | 32.882912  | 2.891243 | -4.989814                  | 0.0                         | 0.0             | 0.271549      | -0.606772  | L-Lysine                             | Alkaloids                | Tyrosine alkaloids    | Phenylethylamines     |
| 2  | Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O | 1.0             | 41.540812  | 2.722000 | -6.059630                  | 1.0                         | 0.0             | 0.250526      | -1.701435  | Dimethylallyl diphosphate            | Carbohydrates            | Nucleosides          | Purine nucleosides    |
| 3  | CC(=O)OCC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)CC#N)[...] | 0.0           | 52.921825  | 0.329071 | -5.473660                  | 0.0                         | 0.0             | 0.267842      | -2.512460  | Geranylgeranyl diphosphate; L-Alanine | Amino acids and Peptides | β-lactams            | Cephalosporins        |


## Visualization

You can use our API to access the bokeh representation of the chemical space and check some features of the molecules:

```python
from deepmol_models import bokeh_plot
bokeh_plot(results, "Solubility", additional_labels=["Pathways", "Superclass", "Class"])
```

![bokeh](https://github.com/BioSystemsUM/deepmol_case_studies/blob/main/example_bokeh.gif)
