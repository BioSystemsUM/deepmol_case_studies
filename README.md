# Case studies for the DeepMol publication

## AutoML experiments

AutoML experiments can be found at **[here](scripts/tdc/)**.

## Evaluate TDC commons benchmark datasets

To train and evaluate DeepMol models:

```python
from dcs.evaluation import get_results

results = get_results(tdc_dataset_name="Bioavailability_Ma", pipeline="bioavailability_optimal")
```

The **tdc_dataset_name** parameter is used to download the TDC commons benchmark datasets. Available datasets:
- "AMES"
- "BBB_Martins"
- "Bioavailability_Ma"
- "Caco2_Wang"
- "Clearance_Hepatocyte_AZ"
- "Clearance_Microsome_AZ"
- "HIA_Hou"
- "Pgp_Broccatelli"
- "Solubility_AqSolDB"
- "Lipophilicity_AstraZeneca"
- "VDss_Lombardo"
- "CYP2C9_Veith"
- "CYP2D6_Veith"
- "CYP3A4_Veith"
- "CYP2C9_Substrate_CarbonMangels"
- "CYP2D6_Substrate_CarbonMangels"
- "CYP3A4_Substrate_CarbonMangels"
- "DILI"
- "Half_Life_Obach"
- "hERG"
- "LD50_Zhu"
- "PPBR_AZ"

While the **pipeline** parameter is for internal pipeline loading. The pipelines are listed according to the paper, where there are the pipelines created based on the first AutoML experiment and the ones that were further optimized (optimal). Available pipelines:

- "ames"
- "bbb"
- "bioavailability" 
- "bioavailability_optimal"
- "caco"
- "clearance_hepatocyte"
- "clearance_microsome"
- "hia"
- "pgp"
- "solubility"
- "lipophilicity"
- "lipophilicity_optimal"
- "vdss"
- "cyp2c9"
- "cyp2d6"
- "cyp3a4" 
- "cyp2c9_substrate"
- "cyp2d6_substrate"
- "cyp3a4_substrate"
- "dili"
- "half_life"
- "herg"
- "hia"
- "ld50"
- "ppbr"

If intended, the default pipelines (all but the optimal) for each dataset can be called as follows:

```python
from dcs.evaluation import get_results

results = get_results(tdc_dataset_name="Bioavailability_Ma", pipeline="bioavailability")
```

