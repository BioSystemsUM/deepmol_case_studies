from tdc.benchmark_group import admet_group


get_benchmark_group = {
    'Caco2_Wang': admet_group(path='data_admet/'),
    'PAMPA_NCATS': admet_group(path='data_admet/'),
    'HIA_Hou': admet_group(path='data_admet/'),
    'Pgp_Broccatelli': admet_group(path='data_admet/'),
    'Bioavailability_Ma': admet_group(path='data_admet/'),
    'Lipophilicity_AstraZeneca': admet_group(path='data_admet/'),
    'Solubility_AqSolDB': admet_group(path='data_admet/'),
    'HydrationFreeEnergy_FreeSolv': admet_group(path='data_admet/'),
    'BBB_Martins': admet_group(path='data_admet/'),
    'PPBR_AZ': admet_group(path='data_admet/'),
    'VDss_Lombardo': admet_group(path='data_admet/'),
    'CYP2C19_Veith': admet_group(path='data_admet/'),
    'CYP2D6_Veith': admet_group(path='data_admet/'),
    'CYP3A4_Veith': admet_group(path='data_admet/'),
    'CYP1A2_Veith': admet_group(path='data_admet/'),
    'CYP2C9_Veith': admet_group(path='data_admet/'),
    'CYP2C9_Substrate_CarbonMangels': admet_group(path='data_admet/'),
    'CYP2D6_Substrate_CarbonMangels': admet_group(path='data_admet/'),
    'CYP3A4_Substrate_CarbonMangels': admet_group(path='data_admet/'),
    'Half_Life_Obach': admet_group(path='data_admet/'),
    'Clearance_Hepatocyte_AZ': admet_group(path='data_admet/'),
}
