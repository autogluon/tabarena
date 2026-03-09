import argparse
import json
import os

import pandas as pd
from pymfe.mfe import MFE
import numpy as np
import tensorflow as tf
from tabpfn import TabPFNClassifier


def get_pymfe_metafeatures(feature):
    pymfe = MFE()
    pymfe.fit(np.array(feature))
    metafeatures = pymfe.extract()
    return metafeatures


def get_pandas_metafeatures(feature_df, featurename):
    feature_pandas_description = feature_df.describe(include="all")
    feature_pandas_description = check_and_complete_pandas_description(feature_pandas_description)
    feature_metadata = {
        "feature - count": feature_pandas_description.loc["count"].values[0],
        "feature - unique": feature_pandas_description.loc["unique"].values[0],
        "feature - top": feature_pandas_description.loc["top"].values[0],
        "feature - freq": feature_pandas_description.loc["freq"].values[0],
        "feature - mean": feature_pandas_description.loc["mean"].values[0],
        "feature - std": feature_pandas_description.loc["std"].values[0],
        "feature - min": feature_pandas_description.loc["min"].values[0],
        "feature - 25": feature_pandas_description.loc["25%"].values[0],
        "feature - 50": feature_pandas_description.loc["50%"].values[0],
        "feature - 75": feature_pandas_description.loc["75%"].values[0],
        "feature - max": feature_pandas_description.loc["max"].values[0],
    }
    return feature_metadata


def check_and_complete_pandas_description(feature_pandas_description):
    if "count" not in feature_pandas_description.index:
        feature_pandas_description.loc["count"] = np.nan
    if "unique" not in feature_pandas_description.index:
        feature_pandas_description.loc["unique"] = np.nan
    if "top" not in feature_pandas_description.index:
        feature_pandas_description.loc["top"] = np.nan
    if "freq" not in feature_pandas_description.index:
        feature_pandas_description.loc["freq"] = np.nan
    if "mean" not in feature_pandas_description.index:
        feature_pandas_description.loc["mean"] = np.nan
    if "std" not in feature_pandas_description.index:
        feature_pandas_description.loc["std"] = np.nan
    if "min" not in feature_pandas_description.index:
        feature_pandas_description.loc["min"] = np.nan
    if "25%" not in feature_pandas_description.index:
        feature_pandas_description.loc["25%"] = np.nan
    if "50%" not in feature_pandas_description.index:
        feature_pandas_description.loc["50%"] = np.nan
    if "75%" not in feature_pandas_description.index:
        feature_pandas_description.loc["75%"] = np.nan
    if "max" not in feature_pandas_description.index:
        feature_pandas_description.loc["max"] = np.nan
    return feature_pandas_description


def get_mfe_feature_metadata(feature):
    # mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
    mfe = MFE(groups="all")
    mfe.fit(feature)
    metafeatures = mfe.extract()
    columns = mfe.extract_metafeature_names()
    groups = mfe.parse_by_group(["general", "statistical", "model-based", "info-theory", "landmarking", "complexity", "clustering"], metafeatures)
    return metafeatures, columns, groups


def get_mfe_dataset_metadata(X, y, group):
    # mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
    mfe = MFE(groups=group)
    mfe.fit(X, y)
    metafeatures = mfe.extract()
    columns = mfe.extract_metafeature_names()
    group = mfe.parse_by_group(group, metafeatures)
    return metafeatures, columns, group


def get_tabpfn_embedding(X, y):
    clf = TabPFNClassifier(device="cuda")
    clf.fit(X, y)
    embeddings = clf.get_embeddings(X)
    return embeddings


"""
def get_d2v_metafeatures(dataset_id):
    tf.random.set_seed(0)
    np.random.seed(42)
    dataset_name, datset_split = get_name_and_split_and_save_dataset(dataset_id)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split',
                        help='Select metafeature extraction model (one can take the average of the metafeatures across all 5 splits)',
                        type=int, default=datset_split)
    parser.add_argument('--file', help='Select dataset name', type=str)

    args = parser.parse_args()
    args.file = dataset_name

    def Dataset2VecModel(configuration):
        nonlinearity_d2v = configuration['nonlinearity_d2v']
        # Function F
        units_f = configuration['units_f']
        nhidden_f = configuration['nhidden_f']
        architecture_f = configuration['architecture_f']
        resblocks_f = configuration['resblocks_f']

        # Function G
        units_g = configuration['units_g']
        nhidden_g = configuration['nhidden_g']
        architecture_g = configuration['architecture_g']

        # Function H
        units_h = configuration['units_h']
        nhidden_h = configuration['nhidden_h']
        architecture_h = configuration['architecture_h']
        resblocks_h = configuration['resblocks_h']
        #
        batch_size = configuration["batch_size"]
        trainable = False
        # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
        x = tf.keras.Input(shape=[2], dtype=tf.float32)
        # Number of sampled classes from triplets
        nclasses = tf.keras.Input(shape=[batch_size], dtype=tf.int32, batch_size=1)
        # Number of sampled features from triplets
        nfeature = tf.keras.Input(shape=[batch_size], dtype=tf.int32, batch_size=1)
        # Number of sampled instances from triplets
        ninstanc = tf.keras.Input(shape=[batch_size], dtype=tf.int32, batch_size=1)
        # Encode the predictor target relationship across all instances
        layer = FunctionF(units=units_f, nhidden=nhidden_f, nonlinearity=nonlinearity_d2v, architecture=architecture_f,
                          resblocks=resblocks_f, trainable=trainable)(x)
        # Average over instances
        layer = PoolF(units=units_f)(layer, nclasses[0], nfeature[0], ninstanc[0])
        # Encode the interaction between features and classes across the latent space
        layer = FunctionG(units=units_g, nhidden=nhidden_g, nonlinearity=nonlinearity_d2v, architecture=architecture_g,
                          trainable=trainable)(layer)
        # Average across all instances
        layer = PoolG(units=units_g)(layer, nclasses[0], nfeature[0])
        # Extract the metafeatures
        metafeatures = FunctionH(units=units_h, nhidden=nhidden_h, nonlinearity=nonlinearity_d2v,
                                 architecture=architecture_h, trainable=trainable, resblocks=resblocks_h)(layer)
        # define hierarchical dataset representation model
        dataset2vec = tf.keras.Model(inputs=[x, nclasses, nfeature, ninstanc], outputs=metafeatures)
        return dataset2vec

    # rootdir = os.path.dirname(os.path.realpath(__file__)) + "/dataset2vec"
    rootdir = "src/Metadata/d2v/dataset2vec"
    log_dir = os.path.join(rootdir, "checkpoints", f"searchspace-a/split-0/dataset2vec/vanilla/configuration-0/2025-06-05-18-47-03-578668")
    save_dir = os.path.join(rootdir, "extracted")
    configuration = json.load(open(os.path.join(log_dir, "configuration.txt"), "r"))
    os.makedirs(save_dir, exist_ok=True)

    metafeatures = pd.DataFrame(data=None)
    datasetmf = []

    batch = Batch(configuration['batch_size'])
    dataset = Dataset(args.file, rootdir)
    testsampler = TestSampling(dataset=dataset)

    model = Dataset2VecModel(configuration)

    model.load_weights(os.path.join(log_dir + "/iteration-50/.weights.h5/.weights.h5"))  # , by_name=False, skip_mismatch=False)

    for q in range(10):  # any number of samples
        batch = testsampler.sample_from_one_dataset(batch)
        batch.collect()
        datasetmf.append(model(batch.input).numpy())

    metafeatures = pd.DataFrame(np.vstack(datasetmf).mean(axis=0)[None], index=[args.file])
    return metafeatures
"""
