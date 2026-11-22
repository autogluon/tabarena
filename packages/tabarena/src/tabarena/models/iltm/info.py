from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.iltm.hpo import gen_iltm
from tabarena.models.iltm.model import ILTMModel

# Dataset overlap, self-reported by the authors (autogluon/tabarena#687, issuecomment-7036773755): these TabArena
# and BeyondArena datasets are in iLTM's meta-training or meta-validation collection (the reply does not say which).
# The checkpoint files have not changed since 2025-11-18 (`HF_REVISION` in model.py), so this holds for the hosted
# TabArena results too. Not set as `leakage_evidence` / `leakage_datasets` for now; kept here for when that part of
# the metadata is reworked.
# TabArena (21 of 51): Amazon_employee_access, anneal, Another-Dataset-on-used-Fiat-500, APSFailure, bank-marketing,
#   churn, coil2000_insurance_policies, credit_card_clients_default, diabetes, Diabetes130US, diamonds,
#   GiveMeSomeCredit, heloc, houses, Is-this-a-good-customer, jm1, kddcup09_appetency, MIC, online_shoppers_intention,
#   seismic-bumps, wine_quality
# BeyondArena (as listed by the authors; 38 of them match the 142 BeyondArena datasets, the last two do not):
#   amazon_employee_access, aps_failure, asp_potassco_classification, bad_customer_detection, bank_marketing,
#   body_density_prediction, cardiotocography, churn, coil_2000, covertype, credit_card_clients_default,
#   diabetes_130_us, diamonds, fiat_500, forensic_glass_identification, give_me_some_credit, heart_disease_cleveland,
#   heart_disease_va_long_beach, heloc, houses, indian_liver_patient_dataset, jm1, kdd_cup_09_appetency, kick,
#   ljubljana_primary_tumor, mic, mice_protein_trisomy_discriminant, micro_mass, musk,
#   parkinsons_biomedical_voice_measurements, porto_seguro,
#   rossmann_store_sales, santander_customer_satisfaction, seismic_bumps, south_africa_coronary_heart_disease,
#   student_portuguese_performance, thyroid_discordant, wine_quality,
#   online_shoppers_purchasing_intention_dataset, otto_group_product_classification_challenge
iltm_method_metadata = MethodMetadata.config(
    method="iLTM",
    suite="tabarena-2026-05-13",
    ag_key="TA-ILTM",
    config_default="iLTM_c1_BAG_L1",
    compute="gpu",
    is_bag=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-05-29",
    date_introduced="2025-11",
    reference_url="https://arxiv.org/abs/2511.15941",
    organization="Stanford University",
    license="Apache-2.0",
    display_name="iLTM",
)


iltm_info = ModelInfo(
    model_cls=ILTMModel,
    search_space=gen_iltm,
    method_metadata=iltm_method_metadata,
    pip_extra=("iltm==0.1.7",),
    prefetch_weights=ILTMModel.prefetch_weights,
)
