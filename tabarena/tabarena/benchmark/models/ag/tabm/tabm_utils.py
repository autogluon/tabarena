def get_tabm_auto_batch_size(n_samples: int, n_features) -> int:
    """Adapted version of batch size estimation by Yury Gorishniy, inferred from the choices
    in the TabM paper."""
    if n_samples < 2_800:
        n_samples_batch_size = 32
    elif n_samples < 4_500:
        n_samples_batch_size = 64
    elif n_samples < 6_400:
        n_samples_batch_size = 128
    elif n_samples < 32_000:
        n_samples_batch_size = 256
    else:
        # Cut to avoid OOM on larger datasets.
        n_samples_batch_size = 512


    if n_features < 256:
        n_features_batch_size = 1024
    elif n_features < 512:
        n_features_batch_size = 512
    elif n_features < 1024:
        n_features_batch_size = 256
    elif n_features < 2048:
        n_features_batch_size = 64
    else:
        n_features_batch_size = 32

    return min(n_samples_batch_size, n_features_batch_size)
