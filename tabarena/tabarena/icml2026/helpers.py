import pandas as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error

from autogluon.tabular.models import LGBModel, LinearModel, TabMModel, CatBoostModel
from tabarena.benchmark.models.wrapper.AutoGluon_class import AGSingleBagWrapper
from tabarena.benchmark.models.prep_ag.prep_lr.prep_lr_model import PrepLinearModel
from tabarena.benchmark.models.prep_ag.prep_lgb.prep_lgb_model import PrepLGBModel
from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
from tabarena.benchmark.models.prep_ag.prep_tabpfnv2_5.prep_tabpfnv2_5_model import PrepRealTabPFNv25Model
from tabarena.benchmark.models.prep_ag.prep_tabm.prep_tabm_model import PrepTabMModel
from tabarena.benchmark.models.prep_ag.prep_cat.prep_cat_model import PrepCatBoostModel


from sklearn.preprocessing import TargetEncoder
from category_encoders import LeaveOneOutEncoder


from autogluon.features import ArithmeticFeatureGenerator, OOFTargetEncodingFeatureGenerator, CategoricalInteractionFeatureGenerator, GroupByFeatureGenerator, RandomSubsetTAFC

def run_experiment(X, y, X_test, y_test, model_name, prep_type, target_type):
    if target_type == 'regression':
        metric = root_mean_squared_error
    elif target_type == 'binary':
        metric = lambda x,y: 1-roc_auc_score(x, y)
    else:
        metric = log_loss

    init_params = {"hyperparameters": {}}
    if model_name == "LR" and prep_type == "None":
        model_cls = LinearModel
    elif model_name == "LR" and prep_type != "None":
        model_cls = PrepLinearModel
    elif model_name == "PFN" and prep_type == "None":
        model_cls = RealTabPFNv25Model
    elif model_name == "PFN" and prep_type != "None":
        model_cls = PrepRealTabPFNv25Model
    elif model_name == "TABM" and prep_type == "None":
        model_cls = TabMModel
    elif model_name == "TABM" and prep_type != "None":
        model_cls = PrepTabMModel
    elif model_name == "CAT" and prep_type == "None":
        model_cls = CatBoostModel
    elif model_name == "CAT" and prep_type != "None":
        model_cls = PrepCatBoostModel
    elif model_name in ["GBM", "GBM-OHE"] and prep_type == "None":
        model_cls = LGBModel
        # init_params['hyperparameters'] = {'n_estimators': 10000}
    elif model_name in ["GBM", "GBM-OHE"] and prep_type != "None":
        model_cls = PrepLGBModel
        # init_params = {"problem_type": target_type, "hyperparameters": {'n_estimators': 10000}}
        # if model_name == "GBM-OHE":
        #     init_params["hyperparameters"].update({'max_cat_to_onehot': 1000000})

    # if prep_type == "OOF-TE":
    #     params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {}]], 'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}}
    # else:
    #     params = {}
    params = {}
    if prep_type == "TE":
        agfg = TargetEncoder(random_state=42)
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        X_prep = pd.DataFrame(agfg.fit_transform(X[cat_cols].astype("object"), y), index=X.index, columns=[f'{i}_te' for i in cat_cols])
        X_test_prep = pd.DataFrame(agfg.transform(X_test[cat_cols].astype("object")), index=X_test.index, columns=[f'{i}_te' for i in cat_cols])
        X_prep = pd.concat([X.drop(cat_cols, axis=1), X_prep], axis=1)
        X_test_prep = pd.concat([X_test.drop(cat_cols, axis=1), X_test_prep], axis=1)
        X_used, X_test_used = X_prep, X_test_prep
    elif prep_type == "LOO":
        agfg = LeaveOneOutEncoder(random_state=42)
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        X_prep = pd.DataFrame(agfg.fit_transform(X[cat_cols].astype("object"), y), index=X.index, columns=[f'{i}_loo' for i in cat_cols])
        X_test_prep = pd.DataFrame(agfg.transform(X_test[cat_cols].astype("object")), index=X_test.index, columns=[f'{i}_loo' for i in cat_cols])
        X_prep = pd.concat([X.drop(cat_cols, axis=1), X_prep], axis=1)
        X_test_prep = pd.concat([X_test.drop(cat_cols, axis=1), X_test_prep], axis=1)
        X_used, X_test_used = X_prep, X_test_prep
    elif prep_type == "OOF-TE":
        params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {}]], 'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "OOF-TE-APPEND":
        params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {"passthrough": True}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type in ["CATINT", "CATINT_OOFTE", "CATINT_TE", "CATINT_LOO"]:
        params = {"ag.prep_params": [["CategoricalInteractionFeatureGenerator", {"max_order": 2}], ['OOFTargetEncodingFeatureGenerator', {}]], 'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}}
    elif prep_type in ["CATINT", "CATINT_OOFTE", "CATINT_TE", "CATINT_LOO"]:
        params = {"ag.prep_params": [["CategoricalInteractionFeatureGenerator", {"max_order": 2}], ['OOFTargetEncodingFeatureGenerator', {"passthrough": True}]]}
    elif prep_type in ["CATINT_TE", "CATINT_LOO"]:

        if prep_type == "CATINT_OOFTE":
            pass
        elif prep_type in ["CATINT_TE", "CATINT_LOO"]:
            agfg = CategoricalInteractionFeatureGenerator(target_type=target_type, passthrough=False, max_order=2)
            X_used = pd.concat([X, agfg.fit_transform(X, y)], axis=1)
            X_test_used = pd.concat([X_test, agfg.transform(X_test)], axis=1)
            cat_cols = X_used.select_dtypes(include=['category', 'object']).columns.tolist()
            if prep_type == "CATINT_TE":
                agfg = TargetEncoder(random_state=42)
            elif prep_type == "CATINT_LOO":
                agfg = LeaveOneOutEncoder(random_state=42, handle_missing="value", handle_unknown="value")
            X_prep = pd.DataFrame(agfg.fit_transform(X_used[cat_cols].astype("object"), y), index=X_used.index, columns=[f'{i}_te' for i in cat_cols])
            X_test_prep = pd.DataFrame(agfg.transform(X_test_used[cat_cols].astype("object")), index=X_test_used.index, columns=[f'{i}_te' for i in cat_cols])
            X_used = pd.concat([X_used.drop(cat_cols, axis=1), X_prep], axis=1)
            X_test_used = pd.concat([X_test_used.drop(cat_cols, axis=1), X_test_prep], axis=1)
    elif prep_type == "DROP-CAT":
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        X_used = X.drop(cat_cols, axis=1)
        X_test_used = X_test.drop(cat_cols, axis=1)
    elif prep_type == "DROP-NUM":
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_used = X.drop(num_cols, axis=1)
        X_test_used = X_test.drop(num_cols, axis=1)
    elif prep_type in ["2-ARITHMETIC", "3-ARITHMETIC", "4-ARITHMETIC"]:
        agfg = ArithmeticFeatureGenerator(target_type=target_type, random_state=42, max_order=int(prep_type[0]), passthrough=True, max_new_feats=1500)
        X_used = agfg.fit_transform(X, y)
        X_test_used = agfg.transform(X_test)
    elif prep_type == "RSTAFC":
        agfg = RandomSubsetTAFC(target_type=target_type, passthrough=True)
        X_used = agfg.fit_transform(X, y)
        X_test_used = agfg.transform(X_test)

    else:
        X_used, X_test_used = X, X_test

    # model = model_cls(**init_params)
    # print(X_used.iloc[0])
    model = AGSingleBagWrapper(model_cls,
                                params, 
                                problem_type=target_type, 
                                eval_metric='roc_auc' if target_type=='binary' else ('rmse' if target_type=='regression' else 'log_loss'), 
                                fit_kwargs= {'num_bag_folds': 8, 'verbosity': 0},
                                )
    model.fit(X=X_used, y=y)
    if target_type == 'regression':
        preds = model.predict(X_test_used)
    elif target_type == 'binary':
        preds = model.predict_proba(X_test_used)
        if isinstance(preds, pd.DataFrame) and preds.shape[1] == 2:
            preds = preds.iloc[:, 1]
        elif isinstance(preds, np.ndarray) and preds.shape[1] == 2:
            preds = preds[:, 1]
    else:
        preds = model.predict_proba(X_test_used)
    score = metric(y_test, preds)

    return preds, score, X_used
# Dataset: quick_test, Model: GBM, Prep: DROP-NUM (shape=[(750, 1)]), Performance: 0.7785
# No path specified. Models will be saved in: "AutogluonModels/ag-20260120_230522-001"
# Dataset: quick_test, Model: GBM, Prep: OOF-TE (shape=[(750, 6)]), Performance: 0.3900
# No path specified. Models will be saved in: "AutogluonModels/ag-20260120_230529-001"
# Dataset: quick_test, Model: GBM, Prep: OOF-TE-APPEND (shape=[(750, 7)]), Performance: 0.3900
