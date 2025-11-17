from .context import BenchmarkContext, construct_context
from ..loaders import Paths

# s3_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_11_14/'
s3_prefix = "https://tabrepo.s3.us-west-2.amazonaws.com/contexts/2023_11_14/"

# 244 datasets sorted largest to smallest
datasets = [
    # Commented out due to excessive size
    # "dionis",  # Cumulative Size: 4.5 TB (244 datasets)
    # "KDDCup99",
    # "Airlines_DepDelay_10M",
    # "Kuzushiji-49",
    # "pokerhand",
    # "sf-police-incidents",
    # "helena",  # Cumulative Size: 1.0 TB (238 datasets)
    # "covertype",
    # "Devnagari-Script",
    # "Higgs",
    # "walking-activity",
    # "spoken-arabic-digit",
    # "GTSRB-HOG01",
    # "GTSRB-HOG02",
    # "GTSRB-HOG03",
    # "GTSRB-HueHist",

    "porto-seguro",  # Cumulative Size: 455 GB (228 datasets)  # 211 datasets (succeeded)
    "airlines",
    "ldpa",
    "albert",
    "tamilnadu-electricity",
    "fars",
    "Buzzinsocialmedia_Twitter",
    "nyc-taxi-green-dec-2016",
    "Fashion-MNIST",
    "Kuzushiji-MNIST",
    "mnist_784",
    # "CIFAR_10",  # Failed
    "volkert",  # 200 datasets (succeeded)
    "Yolanda",
    "letter",
    "kr-vs-k",
    "kropt",
    "MiniBooNE",
    "shuttle",
    "jannis",
    "numerai28_6",
    "Diabetes130US",
    "Run_or_walk_information",
    "APSFailure",
    "kick",
    "Allstate_Claims_Severity",
    "Traffic_violations",
    "black_friday",
    "connect-4",  # Cumulative Size: 107 GB (200 datasets)
    "isolet",
    "adult",
    "okcupid-stem",
    "electricity",
    "bank-marketing",
    # "KDDCup09-Upselling",  # Failed
    # "one-hundred-plants-margin",  # Failed
    # "KDDCup09_appetency",  # Failed
    "jungle_chess_2pcs_raw_endgame_complete",
    "2dplanes",
    "fried",
    "Click_prediction_small",  # 175 datasets (succeeded)
    "nomao",
    "Amazon_employee_access",
    "pendigits",
    "microaggregation2",
    "artificial-characters",
    "robert",
    "houses",
    "Indian_pines",
    "diamonds",
    # "guillermo",  # Failed
    # "riccardo",  # Failed
    # "MagicTelescope",  # Failed
    # "nursery",  # Failed  # Cumulative Size: 50 GB (175 datasets)
    "har",
    "texture",
    "fabert",
    "optdigits",
    "mozilla4",
    "volcanoes-b2",
    "eeg-eye-state",
    "volcanoes-b1",
    "OnlineNewsPopularity",
    "volcanoes-b6",
    "dilbert",
    "volcanoes-b5",
    "GesturePhaseSegmentationProcessed",
    "ailerons",
    "volcanoes-d1",
    "volcanoes-d4",
    "mammography",
    "PhishingWebsites",
    "satimage",
    "jm1",
    "first-order-theorem-proving",
    "kdd_internet_usage",
    "eye_movements",
    "wine-quality-white",
    "delta_elevators",
    "mc1",
    "led24",
    "visualizing_soil",
    "house_16H",
    "SpeedDating",
    "bank32nh",
    "bank8FM",
    "cpu_act",
    "cpu_small",
    "kin8nm",
    "puma32H",
    "puma8NH",
    "collins",
    "house_sales",
    "page-blocks",
    "ringnorm",
    "twonorm",
    "delta_ailerons",
    "wind",
    "wall-robot-navigation",
    "elevators",
    "cardiotocography",
    "philippine",
    "pc2",
    "mfeat-factors",
    # "christine",  # Failed
    "phoneme",
    "sylvine",
    "Satellite",
    "pol",
    "churn",
    "wilt",
    "spambase",
    "segment",
    "waveform-5000",
    # "hypothyroid",  # Failed
    "semeion",
    "hiva_agnostic",
    "ada",
    # "yeast",  # Failed
    "Brazilian_houses",
    "steel-plates-fault",
    "pollen",
    "Bioresponse",  # 100 datasets (succeeded)
    "soybean",
    "Internet-Advertisements",
    "topo_2_1",
    "yprop_4_1",
    "UMIST_Faces_Cropped",
    "madeline",  # Cumulative Size: 8.7 GB (100 datasets)
    "micro-mass",
    "gina",
    "jasmine",
    "splice",
    "dna",
    "wine-quality-red",
    "cnae-9",
    "colleges",
    "madelon",
    "ozone-level-8hr",
    "MiceProtein",
    "volcanoes-a2",
    "volcanoes-a3",
    "Titanic",
    "wine_quality",
    "volcanoes-a4",
    "kc1",
    # "eating",  # Failed
    "car",
    # "QSAR-TID-10980",  # Failed
    # "QSAR-TID-11",  # Failed
    "pbcseq",
    "volcanoes-e1",
    "autoUniv-au6-750",
    # "Santander_transaction_value",  # Failed
    "SAT11-HAND-runtime-regression",
    "GAMETES_Epistasis_2-Way_20atts_0_1H_EDM-1_1",
    "GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1",
    "GAMETES_Epistasis_2-Way_20atts_0_4H_EDM-1_1",
    "GAMETES_Epistasis_3-Way_20atts_0_2H_EDM-1_1",
    "GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM-2_001",
    "GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM-2_001",
    "autoUniv-au7-1100",
    "pc3",
    "Mercedes_Benz_Greener_Manufacturing",
    "OVA_Prostate",
    "OVA_Endometrium",
    "OVA_Kidney",
    "OVA_Lung",
    "OVA_Ovary",
    "pc4",
    # "OVA_Breast",  # Failed
    "OVA_Colon",
    "abalone",
    "LED-display-domain-7digit",
    "analcatdata_dmft",
    "cmc",
    "colleges_usnews",
    # "anneal",  # Failed
    "baseball",
    "hill-valley",
    "space_ga",
    "parity5_plus_5",
    "pc1",
    "eucalyptus",
    "qsar-biodeg",
    "synthetic_control",
    "fri_c0_1000_5",
    "fri_c1_1000_50",
    "fri_c2_1000_25",
    "fri_c3_1000_10",
    "fri_c3_1000_25",
    "autoUniv-au1-1000",
    "credit-g",
    "vehicle",
    "analcatdata_authorship",
    "tokyo1",
    "quake",
    "kdd_el_nino-small",
    "diabetes",  # Cumulative Size: 1.0 GB (30 datasets)
    "blood-transfusion-service-center",
    "us_crime",
    "Australian",
    "autoUniv-au7-700",
    "ilpd",
    "balance-scale",
    "arsenic-female-bladder",
    "climate-model-simulation-crashes",
    "cylinder-bands",
    "meta",
    "house_prices_nominal",
    "kc2",
    "rmftsa_ladata",
    "boston_corrected",
    "fri_c0_500_5",
    "fri_c2_500_50",
    "fri_c3_500_10",
    "fri_c4_500_100",
    "no2",
    "pm10",
    "dresses-sales",
    "fri_c3_500_50",
    "Moneyball",
    "socmob",
    "MIP-2016-regression",
    "sensory",
    "boston",
    "arcene",
    "tecator",
]
folds = [0, 1, 2]
local_prefix = "2023_11_14"
date = "2023_11_14"
metadata_join_column = "tid"

configs_prefix = Paths.data_root / 'configs'
configs = [
    f'{configs_prefix}/configs_catboost.json',
    f'{configs_prefix}/configs_fastai.json',
    f'{configs_prefix}/configs_lightgbm.json',
    f'{configs_prefix}/configs_nn_torch.json',
    f'{configs_prefix}/configs_xgboost.json',
    f'{configs_prefix}/configs_rf.json',
    f'{configs_prefix}/configs_xt.json',
    f'{configs_prefix}/configs_ftt.json',
    f'{configs_prefix}/configs_lr.json',
    f'{configs_prefix}/configs_knn.json',
    f'{configs_prefix}/configs_tabpfn.json',
]

kwargs = dict(
    local_prefix=local_prefix,
    s3_prefix=s3_prefix,
    folds=folds,
    date=date,
    task_metadata="task_metadata.csv",
    metadata_join_column=metadata_join_column,
    configs_hyperparameters=configs,
)


D244_F3_C1530: BenchmarkContext = construct_context(
    name="D244_F3_C1530",
    description="Large-scale Benchmark on 244 datasets and 3 folds (330 GB, 211 datasets)",
    datasets=datasets,
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_200: BenchmarkContext = construct_context(
    name="D244_F3_C1530_200",
    description="Large-scale Benchmark on 244 datasets and 3 folds (120 GB, 200 smallest datasets)",
    datasets=datasets[-200:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_175: BenchmarkContext = construct_context(
    name="D244_F3_C1530_175",
    description="Large-scale Benchmark on 244 datasets and 3 folds (57 GB, 175 smallest datasets)",
    datasets=datasets[-175:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_100: BenchmarkContext = construct_context(
    name="D244_F3_C1530_100",
    description="Large-scale Benchmark on 244 datasets and 3 folds (9.5 GB, 100 smallest datasets)",
    datasets=datasets[-100:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_30: BenchmarkContext = construct_context(
    name="D244_F3_C1530_30",
    description="Large-scale Benchmark on 244 datasets and 3 folds (1.1 GB, 30 smallest datasets)",
    datasets=datasets[-30:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_10: BenchmarkContext = construct_context(
    name="D244_F3_C1530_10",
    description="Large-scale Benchmark on 244 datasets and 3 folds (220 MB, 10 smallest datasets)",
    datasets=datasets[-10:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_3: BenchmarkContext = construct_context(
    name="D244_F3_C1530_3",
    description="Large-scale Benchmark on 244 datasets and 3 folds (33 MB, 3 smallest datasets)",
    datasets=datasets[-3:],
    has_baselines=True,
    **kwargs,
)
