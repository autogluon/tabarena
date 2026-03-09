from __future__ import annotations

from tabarena.benchmark.feature_selection_methods.ag.accuracy.Accuracy import Accuracy
from tabarena.benchmark.feature_selection_methods.ag.anova.ANOVA import ANOVA
from tabarena.benchmark.feature_selection_methods.ag.boruta.boruta import Boruta
from tabarena.benchmark.feature_selection_methods.ag.cart.CART import CART
from tabarena.benchmark.feature_selection_methods.ag.cfs.CFS import CFS
from tabarena.benchmark.feature_selection_methods.ag.chi2.Chi2 import Chi2
from tabarena.benchmark.feature_selection_methods.ag.cife.CIFE import CIFE
from tabarena.benchmark.feature_selection_methods.ag.cmim.CMIM import CMIM
from tabarena.benchmark.feature_selection_methods.ag.consistency.Consistency import Consistency
from tabarena.benchmark.feature_selection_methods.ag.disr.DISR import DISR
from tabarena.benchmark.feature_selection_methods.ag.elastic_net.ElasticNet import ElasticNet
from tabarena.benchmark.feature_selection_methods.ag.enumeration.Enumeration import Enumeration
from tabarena.benchmark.feature_selection_methods.ag.fcbf.FCBF import FCBF
from tabarena.benchmark.feature_selection_methods.ag.fisher_score.FisherScore import FisherScore
from tabarena.benchmark.feature_selection_methods.ag.gain_ratio.GainRatio import GainRatio
from tabarena.benchmark.feature_selection_methods.ag.gini.Gini import Gini
from tabarena.benchmark.feature_selection_methods.ag.group_lasso.GroupLasso import GroupLasso
from tabarena.benchmark.feature_selection_methods.ag.impurity.Impurity import Impurity
from tabarena.benchmark.feature_selection_methods.ag.information_gain.InformationGain import InformationGain
from tabarena.benchmark.feature_selection_methods.ag.interact.INTERACT import INTERACT
from tabarena.benchmark.feature_selection_methods.ag.jmi.JMI import JMI
from tabarena.benchmark.feature_selection_methods.ag.laplacian_score.LaplacianScore import LaplacianScore
from tabarena.benchmark.feature_selection_methods.ag.lasso.Lasso import Lasso
from tabarena.benchmark.feature_selection_methods.ag.llm_select.LLMSelect import LLMSelect
from tabarena.benchmark.feature_selection_methods.ag.ls_flip.ls_flip import LocalSearchFeatureSelector_Flip
from tabarena.benchmark.feature_selection_methods.ag.ls_flipswap.ls_flipswap import LocalSearchFeatureSelector_FlipSwap
from tabarena.benchmark.feature_selection_methods.ag.mafese.MAFESE import MAFESE
from tabarena.benchmark.feature_selection_methods.ag.markov_blanket.MarkovBlanket import MarkovBlanket
from tabarena.benchmark.feature_selection_methods.ag.mcfs.MCFS import MCFS
from tabarena.benchmark.feature_selection_methods.ag.metafs.MetaFS import MetaFS
from tabarena.benchmark.feature_selection_methods.ag.mi.MI import MI
from tabarena.benchmark.feature_selection_methods.ag.mrmr.mRMR import mRMR
from tabarena.benchmark.feature_selection_methods.ag.one_r.OneR import OneR
from tabarena.benchmark.feature_selection_methods.ag.original.Original import Original
from tabarena.benchmark.feature_selection_methods.ag.pearson_correlation.PearsonCorrelation import PearsonCorrelation
from tabarena.benchmark.feature_selection_methods.ag.random.Random import Random
from tabarena.benchmark.feature_selection_methods.ag.relieff.ReliefF import ReliefF
from tabarena.benchmark.feature_selection_methods.ag.rf_importance.RFImportance import RFImportance
from tabarena.benchmark.feature_selection_methods.ag.sbe.SBE import SBE
from tabarena.benchmark.feature_selection_methods.ag.select_k_best_f.select_k_best_f import Select_k_Best_F
from tabarena.benchmark.feature_selection_methods.ag.sfbe.SFBE import SFBE
from tabarena.benchmark.feature_selection_methods.ag.sffs.SFFS import SFFS
from tabarena.benchmark.feature_selection_methods.ag.sfs.SFS import SFS
from tabarena.benchmark.feature_selection_methods.ag.spectral_fs.Spectral import Spectral
from tabarena.benchmark.feature_selection_methods.ag.symmetrical_uncertainty.SymmetricalUncertainty import (
    SymmetricalUncertainty,
)
from tabarena.benchmark.feature_selection_methods.ag.t_test.tTest import tTest

__all__ = [
    "ANOVA",
    "CART",
    "CFS",
    "CIFE",
    "CMIM",
    "DISR",
    "FCBF",
    "INTERACT",
    "JMI",
    "MAFESE",
    "MCFS",
    "MI",
    "SBE",
    "SFBE",
    "SFFS",
    "SFS",
    "Accuracy",
    "Boruta",
    "Chi2",
    "Consistency",
    "ElasticNet",
    "EnumerationFeatureSelector",
    "FisherScore",
    "GainRatio",
    "Gini",
    "GroupLasso",
    "Impurity",
    "InformationGain",
    "LLMSelect",
    "LaplacianScore",
    "Lasso",
    # Other methods
    "LocalSearchFeatureSelector_Flip",
    "LocalSearchFeatureSelector_FlipSwap",
    "MarkovBlanket",
    "MetaFS",
    "OneR",
    "Original",
    "PearsonCorrelation",
    "RFImportance",
    "RandomFS",
    "ReliefF",
    "Select_k_Best_F",
    "Spectral",
    "SymmetricalUncertainty",
    "mRMR",
    # Chosen Filter Methods
    "tTest",
]
