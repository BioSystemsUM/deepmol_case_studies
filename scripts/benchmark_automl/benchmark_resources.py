import copy
import os
import tracemalloc
from hurry.filesize import size
import datetime
import time
from deepmol.standardizer import ChEMBLStandardizer, BasicStandardizer, CustomStandardizer
import numpy as np
import pandas as pd
from deepmol.compound_featurization import *
from deepmol.scalers import *
from deepmol.feature_selection import *
from deepmol.splitters import *
from deepmol.imbalanced_learn import *
from deepmol.loaders import CSVLoader, SDFLoader

def benchmark_methods(methods, dataset_path, smiles_field, label, dataset_name, output_file):

    from deepmol.loaders import CSVLoader

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    for method in methods:
        tracemalloc.start()
        start = time.time()
        method.fit_transform(d1)
        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        results = pd.concat((results, 
                                pd.DataFrame({"number of molecules": [len(d1.mols)],
                                              "method": [method.__class__.__name__], 
                                                "dataset": [dataset_name], 
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv(output_file, index=False)


def benchmark_standardizers(dataset_path, smiles_field, label, dataset_name, output_file):

    methods = [BasicStandardizer(), CustomStandardizer(), ChEMBLStandardizer()]
        
    benchmark_methods(methods, dataset_path, smiles_field, label, dataset_name, output_file)

        
def benchmark_featurizers(dataset_path, smiles_field, label, dataset_name, output_file):

    methods = [TwoDimensionDescriptors(), MorganFingerprint(), AtomPairFingerprint(), LayeredFingerprint(), RDKFingerprint(), MACCSkeysFingerprint(),
               WeaveFeat(), ConvMolFeat(), MolGraphConvFeat(),
        SmileImageFeat(), SmilesSeqFeat(), PagtnMolGraphFeat(), DMPNNFeat(), MATFeat(), SmilesOneHotEncoder()]
    
    methods = [Mol2Vec()]
    

    benchmark_methods(methods, dataset_path, smiles_field, label, dataset_name, output_file)

def benchmark_3d_featurizers(dataset_path, smiles_field, label, dataset_name, output_file):

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Load data from CSV file
    loader = SDFLoader(dataset_path=dataset_path,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset()

    max_atoms = max([mol.GetNumAtoms() for mol in d1.mols])
    print(max_atoms)

    methods = [All3DDescriptors(mandatory_generation_of_conformers=False), CoulombFeat(max_atoms = max_atoms, generate_conformers=False), CoulombEigFeat(max_atoms = max_atoms, generate_conformers=False)]

    for method in methods:
        tracemalloc.start()
        start = time.time()
        method.featurize(d1)
        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        results = pd.concat((results, 
                                pd.DataFrame({"number of molecules": [len(d1.mols)],
                                              "method": [method.__class__.__name__], 
                                                "dataset": [dataset_name], 
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv(output_file, index=False)

def benchmark_scalers(dataset_path, smiles_field, label, dataset_name, output_file):

    methods = [StandardScaler(), RobustScaler(), PowerTransformer(), MinMaxScaler(), MaxAbsScaler(), Normalizer(), 
        Binarizer(), QuantileTransformer()]

    from deepmol.loaders import CSVLoader

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    d1 = TwoDimensionDescriptors().fit_transform(d1)

    for method in methods:
        tracemalloc.start()
        start = time.time()
        method.fit_transform(d1)
        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        results = pd.concat((results, 
                                pd.DataFrame({"number of molecules": [len(d1.mols)],
                                              "method": [method.__class__.__name__], 
                                                "dataset": [dataset_name], 
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv(output_file, index=False)

def benchmark_unbalanced_learn(dataset_path, smiles_field, label, dataset_name, output_file):

    methods = [RandomOverSampler(), SMOTE(), ClusterCentroids(), RandomUnderSampler()]
    methods = [ClusterCentroids(), RandomUnderSampler()]

    from deepmol.loaders import CSVLoader

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    d1 = TwoDimensionDescriptors().fit_transform(d1)
    positive_class = np.sum(d1.y == 1)
    negative_class = np.sum(d1.y == 0)

    for method in methods:
        
        try:
            d2 = copy.deepcopy(d1)
            tracemalloc.start()
            start = time.time()
            d2 = method.sample(d2)
            positive_class_2 = np.sum(d2.y == 1)
            negative_class_2 = np.sum(d2.y == 0)
            end = time.time()
            print("Time spent: ", end - start)
            print("Memory needed: ", tracemalloc.get_traced_memory())
            positive_class = np.sum(d1.y == 1)
            negative_class = np.sum(d1.y == 0)
            results = pd.concat((results, 
                                    pd.DataFrame({"number of molecules start": [f"positive class: {positive_class}, negative class: {negative_class}"],
                                                "number of molecules end": [f"positive class: {positive_class_2}, negative class: {negative_class_2}"],
                                                "method": [method.__class__.__name__], 
                                                    "dataset": [dataset_name], 
                                                    "time": [str(datetime.timedelta(seconds=end - start))], 
                                                    "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                    ignore_index=True, axis=0)
            tracemalloc.stop()

            results.to_csv(output_file, index=False)
        except:
            pass

def benchmark_feature_selectors(dataset_path, smiles_field, label, dataset_name, output_file):
    methods = [KbestFS(), LowVarianceFS(threshold=0.1), PercentilFS(), SelectFromModelFS(), BorutaAlgorithm()
]

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    d1 = MorganFingerprint().fit_transform(d1)

    for method in methods:
        tracemalloc.start()
        start = time.time()
        d1_copy = copy.deepcopy(d1)
        method.fit_transform(d1_copy)
        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        results = pd.concat((results, 
                                pd.DataFrame({"number of molecules": [len(d1.mols)],
                                              "method": [method.__class__.__name__], 
                                                "dataset": [dataset_name], 
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv(output_file, index=False)


def benchmark_splitters(dataset_path, smiles_field, label, dataset_name, output_file):

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    methods = [RandomSplitter(), SingletaskStratifiedSplitter(), SimilaritySplitter(), ScaffoldSplitter(), \
    ButinaSplitter()]

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    for method in methods:
        tracemalloc.start()
        start = time.time()
        train_dataset, test_dataset = method.train_test_split(dataset=d1, frac_train=0.8)


        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        results = pd.concat((results, 
                                pd.DataFrame({"number of molecules": [len(d1.mols)],
                                              "method": [method.__class__.__name__], 
                                                "dataset": [dataset_name], 
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv(output_file, index=False)

def benchmark_structure_generation(dataset_path, smiles_field, label, dataset_name, output_file):

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    tracemalloc.start()
    start = time.time()
    generate_conformers_to_sdf_file(d1, f"{dataset_path.replace('.csv', '')}.sdf", n_conformations=1, threads=15,max_iterations=3, timeout_per_molecule=60*60)
    end = time.time()
    print("Time spent: ", end - start)
    print("Memory needed: ", tracemalloc.get_traced_memory())
    results = pd.concat((results, 
                            pd.DataFrame({"number of molecules": [len(d1.mols)],
                                            "method": ["3D structure generation"], 
                                            "dataset": [dataset_name], 
                                            "time": [str(datetime.timedelta(seconds=end - start))], 
                                            "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                            ignore_index=True, axis=0)
    tracemalloc.stop()

    results.to_csv(output_file, index=False)






benchmark_standardizers("data_for_benchmark/pgp.csv", "Drug", "Y", "pgp", "standardizers_benchmark.csv")
benchmark_standardizers("data_for_benchmark/CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "standardizers_benchmark.csv")
benchmark_standardizers("data_for_benchmark/DEL.csv", "smiles", "true_labels_R", "del", "standardizers_benchmark.csv")

benchmark_featurizers("data_for_benchmark/pgp.csv", "Drug", "Y", "pgp", "featurizers_benchmark.csv")
benchmark_featurizers("data_for_benchmark/CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "featurizers_benchmark.csv")
benchmark_featurizers("data_for_benchmark/DEL.csv", "smiles", "true_labels_R", "del", "featurizers_benchmark.csv")

benchmark_feature_selectors("data_for_benchmark/pgp.csv", "Drug", "Y", "pgp", "feature_selectors_benchmark.csv")
benchmark_feature_selectors("data_for_benchmark/CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "feature_selectors_benchmark.csv")
benchmark_feature_selectors("data_for_benchmark/DEL.csv", "smiles", "true_labels_R", "del", "feature_selectors_benchmark.csv")

benchmark_splitters("data_for_benchmark/pgp.csv", "Drug", "Y", "pgp", "spliters_benchmark.csv")
benchmark_splitters("data_for_benchmark/CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "spliters_benchmark.csv")
benchmark_splitters("data_for_benchmark/DEL.csv", "smiles", "true_labels_R", "del", "spliters_benchmark.csv")

benchmark_structure_generation("data_for_benchmark/pgp.csv", "Drug", "Y", "pgp", "3d_structure_benchmark.csv")
benchmark_structure_generation("data_for_benchmark/CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "3d_structure_benchmark.csv")
benchmark_structure_generation("data_for_benchmark/DEL.csv", "smiles", "true_labels_R", "del", "3d_structure_benchmark.csv")

benchmark_3d_featurizers("data_for_benchmark/pgp.sdf", "Drug", "Y", "pgp", "3d_features_benchmark.csv")
benchmark_3d_featurizers("data_for_benchmark/CYP2D6_Veith.sdf", "Drug", "Y", "cyp2d6", "3d_features_benchmark.csv")
benchmark_3d_featurizers("data_for_benchmark/DEL.sdf", "smiles", "true_labels_R", "del", "3d_features_benchmark.csv")

benchmark_unbalanced_learn("data_for_benchmark/pgp.csv", "Drug", "Y", "pgp", "unbalanced_learn_benchmark.csv")
benchmark_unbalanced_learn("data_for_benchmark/CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "unbalanced_learn_benchmark.csv")
benchmark_unbalanced_learn("data_for_benchmark/DEL.csv", "smiles", "true_labels_R", "del", "unbalanced_learn_benchmark.csv")