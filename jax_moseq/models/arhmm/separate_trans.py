"""AR-HMM with a separate transition matrix per group.

Observation parameters are shared across all sessions; only the transition
structure is fitted per group. This mirrors ``pyhsmm``'s separate-transition
models, where each group holds an independent deep copy of the transition
distribution, so both ``betas`` and ``pi`` gain a leading group axis.

Sessions carry their group through ``data["group"]``, an integer index per
session. Unlike moseq2-model, which drops sessions whose group is ``n/a`` when
adding data while still pairing the model's state sequences against the full
session list, every session supplied here is modelled; a caller that wants a
session excluded must leave it out of the data.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from jax_moseq.utils import device_put_as_scalar, mixed_map, mixed_map_jit
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.distributions import sample_hmm_stateseq
from jax_moseq.utils.transitions import (
    init_hdp_transitions_by_group,
    resample_hdp_transitions_by_group,
)

from jax_moseq.models.arhmm.gibbs import resample_ar_params, session_log_likelihoods
from jax_moseq.models.arhmm.initialize import init_ar_params, init_hyperparams

na = jnp.newaxis


@mixed_map_jit
def resample_discrete_stateseqs(
    seed, x, mask, group, Ab, Q, pi, *, mixed_map_iters, mixed_map_gpus, **kwargs
):
    """Resample ``z``, each session using its own group's transition matrix."""
    nlags = get_nlags(Ab)

    def sample(seed_i, x_i, mask_i, pi_i):
        lls = session_log_likelihoods(x_i, Ab, Q)
        return sample_hmm_stateseq(seed_i, pi_i, lls, mask_i.astype(float)[nlags:])[1]

    return mixed_map(sample, iters=mixed_map_iters, gpus=mixed_map_gpus)(
        jr.split(seed, mask.shape[0]), x, mask, pi[group]
    )


def resample_model(
    data, seed, states, params, hypparams, states_only=False, verbose=False, **kwargs
):
    """Resample the separate-transition ARHMM."""
    seed = jr.split(seed)[1]
    params = params.copy()
    states = states.copy()

    if not states_only:
        if verbose:
            print("Resampling per-group pi")
        params["betas"], params["pi"] = resample_hdp_transitions_by_group(
            seed, states["z"], data["mask"], data["group"], params["betas"],
            **hypparams["trans_hypparams"]
        )

        if verbose:
            print("Resampling Ab,Q (shared across groups)")
        params["Ab"], params["Q"] = resample_ar_params(
            seed, mask=data["mask"], x=data["x"], z=states["z"],
            **hypparams["ar_hypparams"]
        )

    if verbose:
        print("Resampling z")
    states["z"] = resample_discrete_stateseqs(
        seed, data["x"], data["mask"], data["group"],
        params["Ab"], params["Q"], params["pi"],
    )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
    }


def init_model(
    data=None,
    states=None,
    params=None,
    hypparams=None,
    seed=jr.PRNGKey(0),
    trans_hypparams=None,
    ar_hypparams=None,
    num_groups=None,
    verbose=False,
    **kwargs
):
    """Initialize a separate-transition ARHMM.

    ``num_groups`` defaults to one more than the largest group index present in
    the data, so a caller that labels groups ``0..G-1`` needs not supply it.
    """
    if not (data or states):
        raise ValueError("Must provide either `data` or `states`.")
    if not (hypparams or (trans_hypparams and ar_hypparams)):
        raise ValueError(
            "Must provide either `hypparams` or both `trans_hypparams` "
            "and `ar_hypparams`."
        )

    model = {}
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model["seed"] = seed

    if num_groups is None:
        num_groups = int(jnp.max(data["group"])) + 1

    if hypparams is None:
        if verbose:
            print("Separate-trans ARHMM: Initializing hyperparameters")
        hypparams = init_hyperparams(trans_hypparams, ar_hypparams)
    else:
        hypparams = device_put_as_scalar(hypparams)
    hypparams["trans_hypparams"] = dict(
        hypparams["trans_hypparams"], num_groups=num_groups
    )
    model["hypparams"] = hypparams

    if params is None:
        if verbose:
            print("Separate-trans ARHMM: Initializing parameters")
        trans = hypparams["trans_hypparams"]
        params = {}
        params["betas"], params["pi"] = init_hdp_transitions_by_group(
            seed, trans["num_states"], num_groups,
            trans["alpha"], trans["kappa"], trans["gamma"],
        )
        params["Ab"], params["Q"] = init_ar_params(
            seed, **hypparams["ar_hypparams"]
        )
    model["params"] = params

    if states is None:
        if verbose:
            print("Separate-trans ARHMM: Initializing states")
        states = {
            "z": resample_discrete_stateseqs(
                seed, data["x"], data["mask"], data["group"], **params
            )
        }
    model["states"] = states

    return model
