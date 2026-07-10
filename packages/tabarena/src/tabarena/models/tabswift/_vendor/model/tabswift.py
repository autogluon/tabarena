from __future__ import annotations

from typing import Optional, List
from torch import nn, Tensor


from .learning import ICLearning
import torch
import torch.nn.functional as F
class TabSwift(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabSwift is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    For datasets with more than `max_classes` classes, TabSwift switches to hierarchical classification
    to recursively partition classes into subgroups, forming a multi-level classification tree.

    Parameters
    ----------
    max_classes : int, default=10
        Number of classes that the model supports natively. If the number of classes
        in the dataset exceeds this value, hierarchical classification is used.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer

    col_nhead : int, default=4
        Number of attention heads in the column embedding transformer

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer

    icl_nhead : int, default=4
        Number of attention heads in the in-context learning transformer

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components

    dropout : float, default=0.0
        Dropout probability across all components

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components
    """

    def __init__(
        self,
        max_classes: int = 10,
        max_dim: int=100,
        embed_dim: int = 128,
        proj_dim: int = 100,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        register_tokens: int=64,
        use_headwise_gate: bool = False,
        use_elementwise_gate: bool = False,
    ):
        super().__init__()
        assert proj_dim == max_dim
        self.max_classes = max_classes
        self.max_dim = max_dim
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.proj_dim = proj_dim
        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL
        self.x_linear = nn.Linear(self.proj_dim, icl_dim)
        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            register_tokens=register_tokens,
            use_headwise_gate=use_headwise_gate,
            use_elementwise_gate=use_elementwise_gate,
        )

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset.

        Returns
        -------
        Tensor
            Raw logits of shape (B, T, max_classes), which will be further handled by the training code.
        """

        B, T, H = X.shape
        pad_len = self.max_dim - H

        if pad_len < 0:
            raise ValueError("当前维度 H > 100，无法补齐到100")
        X = X/(H/self.max_dim)
        if pad_len > 0:
            X = F.pad(X, pad=(0, pad_len))
        # print(X.shape)
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Column-wise embedding -> Row-wise interaction
        # representations = self.row_interactor(
        #     self.col_embedder(X, d=d, train_size=None if embed_with_test else train_size), d=d
        # )
        representations = self.x_linear(X)

        # Dataset-wise in-context learning
        out = self.icl_predictor(representations, y_train=y_train)

        return out

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        device: Optional[str | torch.device] = None,
        use_amp: bool = True,
        verbose: bool = False,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        inference_config: InferenceConfig
            Inferenece configuration

        Returns
        -------
        Tensor
            Raw logits or probabilities for test samples of shape (B, test_size, num_classes)
            where test_size = T - train_size
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        # if inference_config is None:
        #     inference_config = InferenceConfig()

        # Column-wise embedding -> Row-wise interaction
        # representations = self.row_interactor(
        #     self.col_embedder(
        #         X,
        #         train_size=None if embed_with_test else train_size,
        #         feature_shuffles=feature_shuffles,
        #         mgr_config=inference_config.COL_CONFIG,
        #     ),
        #     mgr_config=inference_config.ROW_CONFIG,
        # )
        B, T, H = X.shape
        pad_len = 500 - H

        if pad_len < 0:
            raise ValueError("当前维度 H > 500，无法补齐到500")
        H_random_indices = torch.arange(H)
        H_random_indices = torch.randperm(H)
        X = X[:, :, H_random_indices]
        X = X/(H/500)
        X = F.pad(X, pad=(0, pad_len))
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Column-wise embedding -> Row-wise interaction
        # representations = self.row_interactor(
        #     self.col_embedder(X, d=d, train_size=None if embed_with_test else train_size), d=d
        # )
        representations = self.x_linear(X)
        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            # mgr_config=inference_config.ICL_CONFIG,
            device=device,
            use_amp=use_amp,
            verbose=verbose,
        )

        return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        device: Optional[str | torch.device] = None,
        use_amp: bool = True,
        verbose: bool = False,
        if_regression: bool = False,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        device : Optional[str or torch.device], default=None
            Device to use for inference. If None, defaults to torch.device("cuda") if available,
            else torch.device("cpu")

        use_amp : bool, default=True
            Whether to enable automatic mixed precision during inference

        verbose : bool, default=False
            Whether to print detailed information during inference

        Returns
        -------
        Tensor
            Raw logits or probabilities for test samples of shape (B, test_size, num_classes)
            where test_size = T - train_size
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        
        B, T, H = X.shape
        pad_len = self.max_dim - H

        if pad_len < 0:
            raise ValueError("当前维度 H > 100，无法补齐到100")
        X = X/(H/self.max_dim)
        if pad_len > 0:
            X = F.pad(X, pad=(0, pad_len))
        # print(X.shape)
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"
        
        representations = self.x_linear(X)

        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            device=device,
            use_amp=use_amp,
            verbose=verbose,
            if_regression=if_regression
        )

        return out
