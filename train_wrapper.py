from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from glob import glob
import os
import warnings
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("*")
warnings.filterwarnings("ignore")


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [
        os.path.join(data_dir, protein_file) for protein_file in df["protein"]
    ]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df["ligand"]]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def train_model(csv_file):
    model = RandomForestRegressor(
        max_features=0.5097299122960709,
        max_leaf_nodes=2038,
        n_estimators=362,
        n_jobs=-1,
        random_state=12032022,
    )
    ligand_features = pd.read_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + "_ligand_bias_features.csv",
        index_col=0,
    )
    all_embeddings = ligand_features
    labels = pd.read_csv(csv_file)["pk"].to_list()
    model.fit(all_embeddings, labels)
    return model


def predict_model(model_name, csv_file, data_dir):
    model = pickle.load(open(f"data/models/{model_name}.pkl", "rb"))

    ligand_features = pd.read_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + "_ligand_bias_features.csv",
        index_col=0,
    )
    all_embeddings = ligand_features
    predictions = model.predict(all_embeddings)
    _, _, keys, pks = load_csv(csv_file, data_dir)
    return pd.DataFrame({"key": keys, "pred": predictions, "pk": pks})


def generate_rdkit_features(csv_file, data_dir):
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    feature_names = get_rdkit_features_names()
    _, ligand_files, keys, _ = load_csv(csv_file, data_dir)
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(
            delayed(get_rdkit_features)(ligand_file, feature_names)
            for ligand_file in tqdm(ligand_files)
        )
    features = {}
    for pdb, result in zip(keys, results):
        features[pdb] = [result[feature_name] for feature_name in feature_names]
    data = pd.DataFrame(features, index=feature_names).T
    data.to_csv(
        "data/features/"
        + f"{csv_file.split('/')[-1].split('.')[0]}"
        + "_ligand_bias_features.csv"
    )
    return None


def get_rdkit_features_names():
    with open("rdkit_feature_names.txt", "r") as f:
        return f.read().splitlines()


def get_rdkit_features(ligand_file, feature_names):
    descriptors = {d[0]: d[1] for d in Descriptors.descList}
    mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
    features = {}
    for feature_name in feature_names:
        try:
            features[feature_name] = descriptors[feature_name](mol)
        except:
            features[feature_name] = np.nan
    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    args = parser.parse_args()
    if args.train:
        if not os.path.exists(
            "data/features/"
            + f"{args.csv_file.split('/')[-1].split('.')[0]}"
            + "_ligand_bias_features.csv"
        ):
            generate_rdkit_features(args.csv_file, args.data_dir)
        model = train_model(args.csv_file)
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        with open(f"data/models/{args.model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    elif args.predict:
        if not os.path.exists(
            "data/features/"
            + f"{args.val_csv_file.split('/')[-1].split('.')[0]}"
            + "_ligand_bias_features.csv"
        ):
            generate_rdkit_features(args.val_csv_file, args.val_data_dir)
        model = pickle.load(open(f"data/models/{args.model_name}.pkl", "rb"))
        df = predict_model(args.model_name, args.val_csv_file, args.val_data_dir)
        if not os.path.exists("data/results"):
            os.makedirs("data/results")
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
    else:
        raise ValueError("Please specify either --train or --predict")
