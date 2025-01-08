

import pandas as pd
from dcs.models.model import PredictionModel


class TDCModel(PredictionModel):

    results_label = "Prediction"
    
    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        
        # Convert to a dataframe for easier manipulation
        results_df = pd.DataFrame(columns=["ID", "SMILES", self.results_label])
        results_df["ID"] = final_ids
        results_df["SMILES"] = final_smiles_dataset
        results_df[self.results_label] = final_predictions

        return results_df
    
class BBB(TDCModel):

    model_name = "BBB (Blood-Brain Barrier)"
    prediction_type = "Penetrates BBB (1) or does not penetrate BBB (0)"
    description = """
    As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. \n
    Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system. \n
    From MoleculeNet.
    """
    features = """
    - Voting classifier that predicts the ability of a drug to penetrate the Blood-Brain Barrier
    """
    model = "BBB_Martins"
    mode = "classification"
    results_label = "BBB penetration"

class AMES(TDCModel):

    model_name = "AMES Mutagenicity"
    prediction_type = "Mutagenic (1) or not mutagenic (0)"
    description = """
    Mutagenicity means the ability of a drug to induce genetic alterations. \n 
    Drugs that can cause damage to the DNA can result in cell death or other severe adverse effects. \n
    Nowadays, the most widely used assay for testing the mutagenicity of compounds is the Ames experiment which was invented by a professor named Ames. \n
    The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations. \n 
    The dataset is aggregated from four papers.
    """
    features = """
    - Voting classifier that predicts mutagenicity of a drug
    """
    model = "AMES"
    mode = "classification"
    results_label = "AMES Mutagenicity"

class PPBR(TDCModel):

    model_name = "Human plasma protein binding rate (PPBR)"
    prediction_type = "Rate of PPBR expressed in percentage"
    description = """
    The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. \n 
    This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions. \n 
    From a ChEMBL assay deposited by AstraZeneca.
    """
    features = """
    - Regressor that predicts PPBR of a drug
    """
    model = "PPBR_AZ"
    mode = "regression"
    results_label = "Human PPBR"


class VDss(TDCModel):

    model_name = "Volume of Distribution (VD) at steady state"
    prediction_type = "Volume of Distribution expressed in liters per kilogram (L/kg)"
    description = """
    The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. \n
    Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.
    """
    features = """
    - Fast regressor that predicts the Volume of Distribution (VD) at steady state
    """
    model = "VDss_Lombardo"
    mode = "regression"
    results_label = "VDss (L/kg)"

class Caco2(TDCModel):

    model_name = "Caco-2 (Cell Effective Permeability)"
    prediction_type = "Cell Effective Permeability (cm/s)"
    description = """
    The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. \n
    The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue."""
    features = """
    - Graph neural network that predicts the Cell Effective Permeability
    """
    model = "Caco2_Wang"
    mode = "regression"
    results_label = "Cell Effective Permeability (cm/s)"

    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        
        # Convert to a dataframe for easier manipulation
        results_df = pd.DataFrame(columns=["ID", "SMILES", self.results_label])
        results_df["ID"] = final_ids
        results_df["SMILES"] = final_smiles_dataset
        results_df[self.results_label] = final_predictions * -1

        return results_df
    
class HIA(TDCModel):

    model_name = "HIA (Human Intestinal Absorption)"
    prediction_type = "Absorbed (1) or not absorbed (0)"
    description = """
    When a drug is orally administered, it needs to be absorbed from the human gastrointestinal system into the bloodstream of the human body. \n
    This ability of absorption is called human intestinal absorption (HIA) and it is crucial for a drug to be delivered to the target.    """
    features = """
    - Graph neural network that predicts the Human Intestinal Absorption
    """
    model = "HIA_Hou"
    mode = "classification"
    results_label = "Human Intestinal Absorption"

class Bioavailability(TDCModel):

    model_name = "Bioavailability"
    prediction_type = "Bioavailable (1) or not bioavailable (0)"
    description = """
    Oral bioavailability is defined as “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”.    
    """
    features = """
    - Voting classifier that predicts the bioavailability of a compound.
    """
    model = "Bioavailability_Ma"
    mode = "classification"
    results_label = "Bioavailability"

class Lipophilicity(TDCModel):

    model_name = "Lipophilicity"
    prediction_type = "Lipophilicyty log-ratio"
    description = """
    Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. \n 
    High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption. \n
    From MoleculeNet.    """
    features = """
    - Regressor to predict the lipophiicity of a compound
    """
    model = "Lipophilicity_AstraZeneca"
    mode = "regression"
    results_label = "Lipophilicity log-ratio"

class Solubility(TDCModel):

    model_name = "Solubility"
    prediction_type = "Solubility (log mol/L)"
    description = r"""
    Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.    """
    features = """
    - Classifier that predicts the solubility of a compound.
    """
    model = "Solubility_AqSolDB"
    mode = "regression"
    results_label = "Solubility (log mol/L)"

    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        
        # Convert to a dataframe for easier manipulation
        results_df = pd.DataFrame(columns=["ID", "SMILES", self.results_label])
        results_df["ID"] = final_ids
        results_df["SMILES"] = final_smiles_dataset
        results_df[self.results_label] = final_predictions * -1

        return results_df
    
class CYP2D6Inhibition(TDCModel):

    model_name = "CYP P450 2D6 Inhibition"
    prediction_type = "Inhibit (1) or does not inhibit (0)"
    description = """
    The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. \n 
    Specifically, CYP2D6 is primarily expressed in the liver. \n 
    It is also highly expressed in areas of the central nervous system, including the substantia nigra."""
    features = """
    - Classifier that predicts the CYP P450 2D6 inhibition of a compound.
    """
    model = "CYP2D6_Veith"
    mode = "classification"
    results_label = "CYP P450 2D6 Inhibition"

class CYP2C9Inhibition(TDCModel):

    model_name = "CYP P450 2C9 Inhibition"
    prediction_type = "Inhibit (1) or does not inhibit (0)"
    description = """
    The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. \n 
    Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds."""
    features = """
    - Classifier that predicts the CYP P450 2C9 inhibition of a compound.
    """
    model = "CYP2C9_Veith"
    mode = "classification"
    results_label = "CYP P450 2C9 Inhibition"

class CYP3A4Inhibition(TDCModel):

    model_name = "CYP P450 3A4 Inhibition"
    prediction_type = "Inhibit (1) or does not inhibit (0)"
    description = """
    The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. \n
    It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body."""
    features = """
    - Classifier that predicts the CYP P450 3A4 inhibition of a compound.
    """
    model = "CYP3A4_Veith"
    mode = "classification"
    results_label = "CYP P450 3A4 Inhibition"

class CYP2C9Substrate(TDCModel):

    model_name = "CYP2C9 Substrate"
    prediction_type = "Metabolized (1) or does not metabolize (0)"
    description = """
    CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. \n
    Substrates are drugs that are metabolized by the enzyme. TDC used a dataset from [1], which merged information on substrates and nonsubstrates from six publications."""
    features = """
    - Classifier that predicts whether a drug is metabolized by the CYP P450 2C9 enzyme.
    """
    model = "CYP2C9_Substrate_CarbonMangels"
    mode = "classification"
    results_label = "CYP2C9 Substrate"

class CYP2D6Substrate(TDCModel):

    model_name = "CYP2D6 Substrate"
    prediction_type = "Metabolized (1) or does not metabolize (0)"
    description = """
    CYP P450 2D6 plays a major role in the oxidation of both xenobiotic and endogenous compounds. \n
    Substrates are drugs that are metabolized by the enzyme. TDC used a dataset from [1], which merged information on substrates and nonsubstrates from six publications."""
    features = """
    - Classifier that predicts whether a drug is metabolized by the CYP P450 2D6 enzyme.
    """
    model = "CYP2D6_Substrate_CarbonMangels"
    mode = "classification"
    results_label = "CYP2D6 Substrate"

class CYP3A4Substrate(TDCModel):

    model_name = "CYP3A4 Substrate"
    prediction_type = "Metabolized (1) or does not metabolize (0)"
    description = """
    CYP P450 3A4 plays a major role in the oxidation of both xenobiotic and endogenous compounds. \n
    Substrates are drugs that are metabolized by the enzyme. TDC used a dataset from [1], which merged information on substrates and nonsubstrates from six publications."""
    features = """
    - Classifier that predicts whether a drug is metabolized by the CYP P450 3A4 enzyme.
    """
    model = "CYP3A4_Substrate_CarbonMangels"
    mode = "classification"
    results_label = "CYP3A4 Substrate"

class HepatocyteClearance(TDCModel):

    model_name = "Hepatocyte Clearance"
    prediction_type = "Drug hepatocyte clearance (uL.min-1.(10^6 cells)-1)"
    description = """
    Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. \n
    This is a dataset curated from ChEMBL database containing experimental results on intrinsic clearance, deposited from AstraZeneca. \n 
    It contains clearance measures from two experiments types, hepatocyte and microsomes. \n 
    As many studies [2] have shown various clearance outcomes given these two different types, we separate them."""
    features = """
    - Regressor that predicts drug hepatocyte clearance.
    """
    model = "Clearance_Hepatocyte_AZ"
    mode = "regression"
    results_label = "hepatocyte clearance (uL.min-1.(10^6 cells)-1)"

class MicrosomeClearance(TDCModel):

    model_name = "Microsome Clearance"
    prediction_type = "Drug microsome clearance (mL.min-1.g-1)"
    description = """
    Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. \n
    This is a dataset curated from ChEMBL database containing experimental results on intrinsic clearance, deposited from AstraZeneca. \n 
    It contains clearance measures from two experiments types, hepatocyte and microsomes. \n 
    As many studies [2] have shown various clearance outcomes given these two different types, we separate them."""
    features = """
    - Regressor that predicts drug microsome clearance.
    """
    model = "Clearance_Microsome_AZ"
    mode = "regression"
    results_label = "drug microsome clearance (mL.min-1.g-1)"

class hERGBlockers(TDCModel):

    model_name = "hERG blockers"
    prediction_type = "hERG blocker (1) or not blocker (0)"
    description = """
    Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. \n
    Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages."""
    features = """
    - Classifier that predicts hERG blockers.
    """
    model = "hERG"
    mode = "classification"
    results_label = "hERG blocker"


class LD50(TDCModel):

    model_name = "Acute toxicity LD50"
    prediction_type = "LD50 (log(1/(mol/kg)))"
    description = """
    Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug. This dataset is kindly provided by the authors of [1]."""
    features = """
    - Regressor that predicts LD50.
    """
    model = "LD50_Zhu"
    mode = "regression"
    results_label = "LD50 (log(1/(mol/kg)))"

    
