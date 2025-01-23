import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def generate_tsne_molecular_similarities(dataset, smiles_field):
    # Create a function to compute molecular fingerprints
    def compute_fingerprint(smiles):
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        return fingerprint

    # Compute molecular fingerprints for each molecule in the combined dataset
    fingerprints = [compute_fingerprint(smiles) for smiles in dataset.loc[:, smiles_field]]

    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
    # Compute molecular similarities using Tanimoto coefficient
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Apply t-SNE to reduce the dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(similarity_matrix)
    # Separate the embeddings based on the original datasets
    dataset["t-sne 1"] = tsne_embeddings[:, 0]
    dataset["t-sne 2"] = tsne_embeddings[:, 1]

    return dataset

from copy import copy
import bokehmol

from bokeh.plotting import figure, show
from bokeh.io import show, output_notebook
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral10

def generate_bokeh(main_label, dataset, smiles_column, id_field, additional_labels=[]):

    dataset_copy = copy(dataset)

    y = dataset_copy.loc[:, main_label]
    main_label = main_label.replace(" ", "_")

    # Register the dataset with BokehMol
    bokehmol.register_alias()
    output_notebook()

    color_mapper = linear_cmap(field_name=main_label, palette=Spectral10, low=min(y), high=max(y))

    # Create ColumnDataSource
    dataset_copy.columns = [column.replace(" ", "_") for column in dataset_copy.columns]
    from bokeh.models import ColumnDataSource
    source = ColumnDataSource(dataset_copy)

    # Create plot
    plot = figure(
        width=1000, height=1000,
        title="DeepMol Hover",
        background_fill_color="#efefef",
        tools="pan,wheel_zoom",
    )

    tooltips = [("ID", f"@{id_field}")]
    tooltips.append((main_label, f'@{main_label.replace(" ", "_")}'))
    tooltips.extend([(label, f'@{label.replace(" ", "_")}') for label in additional_labels])

    mol_hover = bokehmol.hover.rdkit(
        smiles_column=smiles_column,
        tooltips=tooltips,
        draw_options={
            "comicMode": True
        }
    )

    plot.add_tools(mol_hover)

    plot.circle(
        "t-sne_1", "t-sne_2",
        size=15, line_width=0,
        color=color_mapper, legend_label=main_label,
        source=source
    )

    from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper

    color_bar = ColorBar(color_mapper=LinearColorMapper(palette=Spectral10, low=min(y), high=max(y)), width=8, location=(0,0))
    plot.add_layout(color_bar, 'right')

    # Show plot
    return show(plot)

def bokeh_plot(results, main_label, id_field = "ID", smiles_column = "SMILES", additional_labels=[]):

    results = results[~pd.isna(results.loc[:, main_label])]
    dataset = generate_tsne_molecular_similarities(results, smiles_column)
    generate_bokeh(main_label, dataset, smiles_column, id_field, additional_labels=additional_labels)
