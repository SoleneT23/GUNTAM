from typing import Dict
import torch
import torch.nn.functional as F



def attention_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Compute the attention loss using binary cross-entropy.

    Encourages high attention weights for positive pairs (same particle) and
    low attention weights for negative pairs (different particles).

    Args:
        attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
        pairs1: `[N_pairs]` first indices for evaluated pairs.
        pairs2: `[N_pairs]` second indices for evaluated pairs.
        target: `[N_pairs]` labels in `{+1 (same), -1 (different)}`.

    Note:
        `pairs1`/`pairs2` must be symmetric (include both `(i, j)` and `(j, i)`) and must not
        contain self-pairs `(i, i)`.

    Returns:
        Scalar attention loss tensor
    """

    # Guard: if no pairs, return zero loss
    if pairs1.numel() == 0 or pairs2.numel() == 0:
        # Preserve device from attention tensor
        return torch.tensor(0.0, device=attention_map_bin.device)

    # Direct indexing without masking to surface incorrectly built pairs
    logits = attention_map_bin[pairs1, pairs2]

    # Targets: map {-1, +1} -> {0, 1}
    targets = (target.float() + 1.0) * 0.5

    # Use target magnitude as per-pair weight (PV pairs have target=100 -> higher weight)
    pair_weights = target.abs().float()

    return F.binary_cross_entropy_with_logits(logits, targets, weight=pair_weights, reduction="sum")


def full_attention_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Compute the attention loss using only the last transformer layer's attention.

    Encourages high attention weights for positive pairs (same particle) and
    low attention weights for negative pairs (different particles).
    Compared to attention_loss, this version doesn't use negative pairs, it instead looks
    at all possible pairs not in the positive set as negative. The value of the loss for positive
    and negative pairs is weighted by the number of pairs of each type.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for positive pairs.
            pairs2: `[N_pairs]` second indices for positive pairs.
            target: `[N_pairs]` labels where positives are `+1` (others ignored).

        Note:
            For consistency, positives in (`pairs1`,`pairs2`) should be symmetric (both `(i, j)` and `(j, i)`)
            and should exclude diagonal `(i, i)`.

    Returns:
        Scalar attention loss tensor
    """

    device = attention_map_bin.device
    pos_mask = target > 0
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    # Save per-pair weights (1 for normal, 100 for PV) before shadowing target
    pair_weights_pos = target[pos_mask].abs().float()

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Build a target matrix and a pair-weight matrix
    target_matrix = torch.zeros_like(attention_map_bin, device=device)
    target_matrix[pairs1[pos_mask], pairs2[pos_mask]] = 1.0
    pair_weight_matrix = torch.ones_like(attention_map_bin)
    pair_weight_matrix[pairs1[pos_mask], pairs2[pos_mask]] = pair_weights_pos

    full_mask = torch.ones_like(attention_map_bin, dtype=torch.bool)
    full_mask[num_valid_hits:, :] = False
    full_mask[:, num_valid_hits:] = False

    # Columns corresponding to hits NOT in any positive pair -> mask them out
    inactive_cols = torch.ones(attention_map_bin.shape[0], dtype=torch.bool, device=device)
    inactive_cols[pos_hits] = False
    full_mask[:, inactive_cols] = False

    logits = attention_map_bin[full_mask]
    targets = target_matrix[full_mask]

    pos_weight = 1 / max(pairs1[pos_mask].numel(), 1)
    neg_weight = 1 / max(full_mask.sum().item() - pairs1[pos_mask].numel(), 1)

    # Class weights scaled by per-pair weight (PV pairs contribute more)
    weights = torch.where(targets > 0, pos_weight * pair_weight_matrix[full_mask], neg_weight)

    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")






def top_attention_loss(
    attention_map_bin: torch.Tensor,
    pairs1: torch.Tensor,
    pairs2: torch.Tensor,
    target: torch.Tensor,
    particle_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    return_debug: bool = False,
    hard_negative_fraction: float = 0.2,
) -> torch.Tensor:
    """
    Attention loss using BCE-with-logits.

    Positives:
        sampled same-particle pairs given by pairs1, pairs2, target > 0.

    Negatives:
        pairs of real hits whose particle IDs are different.

    """

    device = attention_map_bin.device
    dtype = attention_map_bin.dtype

    if attention_map_bin.dim() == 3:
        attention_map_bin = attention_map_bin[0]

    seq_len = attention_map_bin.shape[0]

    particle_ids = particle_ids.to(device)

    if particle_ids.dim() == 3:
        particle_ids = particle_ids[0, :, 0]
    elif particle_ids.dim() == 2:
        particle_ids = particle_ids[:, 0]
    else:
        particle_ids = particle_ids.view(-1)

    particle_ids = particle_ids.long()

    padding_mask = padding_mask.to(device)

    if padding_mask.dim() == 2:
        padding_mask = padding_mask[0]
    else:
        padding_mask = padding_mask.view(-1)

    padding_mask = padding_mask.bool()

    real_hit_mask = ~padding_mask

    real_hit_mask = real_hit_mask[:seq_len]
    particle_ids = particle_ids[:seq_len]

    pos_mask = target > 0

    if not torch.any(pos_mask):
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        if return_debug:
            debug_info = {
                "positive_scores": torch.empty(0, device=device, dtype=dtype),
                "negative_scores": torch.empty(0, device=device, dtype=dtype),
                "random_negative_scores": torch.empty(0, device=device, dtype=dtype),
                "hard_negative_scores": torch.empty(0, device=device, dtype=dtype),
                "num_positive_pairs": torch.tensor(0, device=device),
                "num_negative_candidates": torch.tensor(0, device=device),
                "num_selected_negatives": torch.tensor(0, device=device),
            }
            return loss, debug_info

        return loss

    pos_i = pairs1[pos_mask].long().to(device)
    pos_j = pairs2[pos_mask].long().to(device)

    pair_weights_pos = target[pos_mask].abs().float().to(device)

    valid_pos = (
        (pos_i >= 0)
        & (pos_i < seq_len)
        & (pos_j >= 0)
        & (pos_j < seq_len)
    )

    valid_pos = valid_pos & real_hit_mask[pos_i] & real_hit_mask[pos_j]
    valid_pos = valid_pos & torch.isfinite(attention_map_bin[pos_i, pos_j])

    pos_i = pos_i[valid_pos]
    pos_j = pos_j[valid_pos]
    pair_weights_pos = pair_weights_pos[valid_pos]

    if pos_i.numel() == 0:
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        if return_debug:
            debug_info = {
                "positive_scores": torch.empty(0, device=device, dtype=dtype),
                "negative_scores": torch.empty(0, device=device, dtype=dtype),
                "random_negative_scores": torch.empty(0, device=device, dtype=dtype),
                "hard_negative_scores": torch.empty(0, device=device, dtype=dtype),
                "num_positive_pairs": torch.tensor(0, device=device),
                "num_negative_candidates": torch.tensor(0, device=device),
                "num_selected_negatives": torch.tensor(0, device=device),
            }
            return loss, debug_info

        return loss

    pos_scores = attention_map_bin[pos_i, pos_j]
    num_pos = pos_scores.numel()

    valid_pair_mask = real_hit_mask[:, None] & real_hit_mask[None, :]
    same_particle_mask = particle_ids[:, None] == particle_ids[None, :]

    eye = torch.eye(seq_len, dtype=torch.bool, device=device)

    neg_mask = (
        valid_pair_mask
        & (~same_particle_mask)
        & (~eye)
        & torch.isfinite(attention_map_bin)
    )

    neg_scores = attention_map_bin[neg_mask]

    total_num_neg = min(num_pos, neg_scores.numel())

    if total_num_neg == 0:
        random_neg_scores = torch.empty(0, device=device, dtype=dtype)
        hard_neg_scores = torch.empty(0, device=device, dtype=dtype)
        mixed_neg_scores = torch.empty(0, device=device, dtype=dtype)

    else:
        hard_negative_fraction = max(
            0.0,
            min(1.0, float(hard_negative_fraction)),
        )

        num_hard = int(total_num_neg * hard_negative_fraction)
        num_random = total_num_neg - num_hard

        if num_hard > 0:
            hard_neg_scores, hard_indices = torch.topk(
                neg_scores,
                k=num_hard,
                largest=True,
                sorted=False,
            )
        else:
            hard_neg_scores = torch.empty(0, device=device, dtype=dtype)
            hard_indices = torch.empty(0, device=device, dtype=torch.long)

        if num_random > 0:
            remaining_mask = torch.ones(
                neg_scores.numel(),
                dtype=torch.bool,
                device=device,
            )

            if hard_indices.numel() > 0:
                remaining_mask[hard_indices] = False

            remaining_scores = neg_scores[remaining_mask]

            if remaining_scores.numel() == 0:
                random_neg_scores = torch.empty(0, device=device, dtype=dtype)
            else:
                actual_num_random = min(
                    num_random,
                    remaining_scores.numel(),
                )

                random_indices = torch.randperm(
                    remaining_scores.numel(),
                    device=device,
                )[:actual_num_random]

                random_neg_scores = remaining_scores[random_indices]

        else:
            random_neg_scores = torch.empty(0, device=device, dtype=dtype)

        mixed_neg_scores = torch.cat(
            [
                random_neg_scores,
                hard_neg_scores,
            ],
            dim=0,
        )

    logits = torch.cat(
        [
            pos_scores,
            mixed_neg_scores,
        ],
        dim=0,
    )

    targets = torch.cat(
        [
            torch.ones(num_pos, device=device, dtype=dtype),
            torch.zeros(mixed_neg_scores.numel(), device=device, dtype=dtype),
        ],
        dim=0,
    )

    pos_weight = 1.0 / max(num_pos, 1)
    neg_weight = 1.0 / max(mixed_neg_scores.numel(), 1)

    pos_weights = pos_weight * pair_weights_pos.to(dtype)

    neg_weights = torch.full(
        (mixed_neg_scores.numel(),),
        neg_weight,
        device=device,
        dtype=dtype,
    )

    weights = torch.cat(
        [
            pos_weights,
            neg_weights,
        ],
        dim=0,
    )

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weights,
        reduction="sum",
    )

    if return_debug:
        debug_info = {
            "positive_scores": pos_scores.detach(),
            "negative_scores": mixed_neg_scores.detach(),
            "random_negative_scores": random_neg_scores.detach(),
            "hard_negative_scores": hard_neg_scores.detach(),
            "num_positive_pairs": torch.tensor(num_pos, device=device),
            "num_negative_candidates": torch.tensor(
                neg_scores.numel(),
                device=device,
            ),
            "num_selected_negatives": torch.tensor(
                mixed_neg_scores.numel(),
                device=device,
            ),
        }
        return loss, debug_info

    return loss



def attention_next_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Attention loss using cross-entropy for sequential pairs.

    For each hit i in the sequence, if there exists a positive pair (i, i+1),
    we use the attention distribution from hit i as logits and apply cross-entropy
    loss with target = i+1. This encourages the model to attend to the next hit
    in the same particle track.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for pairs.
            pairs2: `[N_pairs]` second indices for pairs.
            target: `[N_pairs]` labels where positives are `+1` used to derive next targets.

        Note:
            Pairs should be symmetric across direction `(i, j)` and `(j, i)` and must not include `(i, i)`.

    """
    device = attention_map_bin.device
    pos_mask = target > 0
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Positive pairs within valid hit range
    sources = pairs1[pos_mask]
    targets = pairs2[pos_mask]
    pair_weights = target[pos_mask].float()  # 1.0 for normal, 100.0 for PV

    # Unique sources among valid hits
    unique_sources = torch.unique(sources)

    # Restrict logits to valid hits (slice the attention map, not the function)
    attention_logits = attention_map_bin[:num_valid_hits, :num_valid_hits]
    masked_logits = attention_logits[unique_sources].clone()  # [S, num_valid_hits]

    selected_targets = torch.full_like(unique_sources, fill_value=num_valid_hits)
    source_weights = torch.ones(unique_sources.numel(), device=device)

    if unique_sources.numel() > 0 and sources.numel() > 0:
        # Build [S, M] match matrix (S=unique sources, M=pairs)
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)  # [S, M]
        targets_row = targets.view(1, -1)

        # Prefer min forward target (t > s); fall back to max backward (t < s).
        fwd_mask = source_eq & (targets > sources).view(1, -1)  # [S, M]
        back_mask = source_eq & (targets <= sources).view(1, -1)  # [S, M]

        fwd_min, _ = torch.min(torch.where(fwd_mask, targets_row, torch.full_like(targets_row, num_valid_hits)), dim=1)
        back_max, _ = torch.max(torch.where(back_mask, targets_row, torch.full_like(targets_row, -1)), dim=1)
        fwd_exists = fwd_mask.any(dim=1)

        selected_targets = torch.where(fwd_exists, fwd_min, back_max)

        # Suppress trajectory partners NOT chosen as selected_targets so they
        # don't compete with true negatives in the cross-entropy softmax.
        other_traj_mask = source_eq & (targets_row != selected_targets.view(-1, 1))  # [S, M]
        s_indices, m_indices = torch.where(other_traj_mask)
        if s_indices.numel() > 0:
            other_traj_cols = torch.zeros(unique_sources.numel(), num_valid_hits, dtype=torch.bool, device=device)
            other_traj_cols[s_indices, targets[m_indices]] = True
            masked_logits[other_traj_cols] = float("-inf")

        # ── Per-source weights ──
        weights_row = pair_weights.view(1, -1).expand(unique_sources.numel(), -1)  # [S, M]
        source_weights = torch.max(torch.where(source_eq, weights_row, torch.zeros_like(weights_row)), dim=1).values  # [S]

    per_sample_loss = F.cross_entropy(masked_logits, selected_targets, reduction="none")
    loss = (per_sample_loss * source_weights).sum()

    return loss


def attention_backward_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Attention loss using cross-entropy for sequential pairs.

    For each hit i in the sequence, if there exists a positive pair (i, i-1),
    we use the attention distribution from hit i as logits and apply cross-entropy
    loss with target = i-1. This encourages the model to attend to the previous hit
    in the same particle track.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for pairs.
            pairs2: `[N_pairs]` second indices for pairs.
            target: `[N_pairs]` labels where positives are `+1` used to derive next targets.

        Note:
            Pairs should be symmetric across direction `(i, j)` and `(j, i)` and must not include `(i, i)`.

    """
    device = attention_map_bin.device
    pos_mask = target > 0
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Positive pairs within valid hit range
    sources = pairs1[pos_mask]
    targets = pairs2[pos_mask]
    pair_weights = target[pos_mask].abs().float()  # 1.0 for normal, 100.0 for PV

    # Unique sources among valid hits
    unique_sources = torch.unique(sources)

    # For each source s: choose
    #  - last backward target: max t where t < s
    #  - else (no backward), next forward target: min t where t > s
    selected_targets = torch.full_like(unique_sources, fill_value=num_valid_hits)

    # Vectorized selection per source: prefer max backward target (< s), else min forward (> s)
    if unique_sources.numel() > 0 and sources.numel() > 0:
        # Build [S, M] match matrix (S=unique sources, M=pairs)
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)  # [S, M]

        # Pair-wise forward/backward masks (per pair, relative to its own source)
        forward_pairs_mask = targets >= sources  # [M]
        backward_pairs_mask = targets < sources  # [M]

        # Broadcast to [S, M]
        fwd_mask = source_eq & forward_pairs_mask.view(1, -1)
        back_mask = source_eq & backward_pairs_mask.view(1, -1)

        targets_row = targets.view(1, -1)

        # For backward: take max target; use sentinel = -1 when absent
        back_candidates = torch.where(back_mask, targets_row, torch.full_like(targets_row, -1))
        back_max, _ = torch.max(back_candidates, dim=1)  # [S]
        back_exists = back_mask.any(dim=1)  # [S]

        # For forward: take min target; use sentinel = num_valid_hits when absent
        fwd_candidates = torch.where(fwd_mask, targets_row, torch.full_like(targets_row, num_valid_hits))
        fwd_min, _ = torch.min(fwd_candidates, dim=1)  # [S]

        # Prefer backward if exists, else forward
        selected_targets = torch.where(back_exists, back_max, fwd_min)
    # else: keep selected_targets as sentinel

    # Restrict logits to valid hits (slice the attention map, not the function)
    attention_logits = attention_map_bin[:num_valid_hits, :num_valid_hits]

    # Gather per-source weight: max pair weight among all pairs originating from each source
    if unique_sources.numel() > 0 and sources.numel() > 0:
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)  # [S, M]
        weights_row = pair_weights.view(1, -1).expand(unique_sources.numel(), -1)  # [S, M]
        source_weights = torch.max(torch.where(source_eq, weights_row, torch.zeros_like(weights_row)), dim=1).values  # [S]
    else:
        source_weights = torch.ones(unique_sources.numel(), device=device)

    per_sample_loss = F.cross_entropy(attention_logits[unique_sources], selected_targets, reduction="none")
    loss = (per_sample_loss * source_weights).sum()

    return loss


def reconstruction_loss(
    reconstructed_particle: torch.Tensor,
    particles_data: torch.Tensor,
    padded_mask: torch.Tensor,
    loss_type: str = "MSE",
) -> Dict[str, torch.Tensor]:
    """
    Compute the reconstruction loss for the particles.

    Computes element-wise loss (MSE or L1) between reconstructed and original
    particle properties for each physical quantity. Excludes padded hits and
    hits without associated particles (pT == 0).

    Expected tensor layout:
    - `reconstructed_particle`: [batch, max_hits, 5] with components [z, eta, sin(phi), cos(phi), pT]
    - `particles_data`:        [batch, max_hits, 4] with components [z, eta, phi, pT]
    - `padded_mask`:           [batch, max_hits] boolean (True means padded)

    Args:
        reconstructed_particle: Reconstructed particle predictions.
        particles_data: Original particle data.
        padded_mask: Boolean mask for padded hits (True for padded).
        loss_type: "MSE" or "L1".

    Returns:
        Dictionary containing reconstruction losses for each particle property:
        {'z': loss_z, 'eta': loss_eta, 'phi': loss_phi, 'pt': loss_pt}
    """
    device = reconstructed_particle.device
    loss_function = None
    if loss_type == "MSE":
        loss_function = F.mse_loss
    elif loss_type == "L1":
        loss_function = F.l1_loss
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}' in reconstruction_loss")

    # Check if there are any valid (non-padded) hits
    # padded_mask is 1 for padded hits, so we need to invert it
    non_padded_mask = ~padded_mask.bool()

    # Filter out hits without associated particles (pT = 0.0 indicates orphan hits)
    # Only hits with true pT > 0 participate in any reconstruction component (z, eta, phi, pT)
    has_particle_mask = particles_data[:, :, 4] > 0.0  # True particle present (index 4 = pT)
    valid_hits_mask = non_padded_mask & has_particle_mask

    if torch.sum(valid_hits_mask) == 0:
        # Return zero losses if no valid hits with particles
        return {
            "z": torch.tensor(0.0, device=device),
            "eta": torch.tensor(0.0, device=device),
            "phi": torch.tensor(0.0, device=device),
            "pt": torch.tensor(0.0, device=device),
        }
    # Safe clamp for pT inversion to avoid inf / NaN

    # Compute component losses
    loss_z_part = loss_function(
        reconstructed_particle[:, :, 0][valid_hits_mask],
        particles_data[:, :, 1][valid_hits_mask],
        reduction="sum",
    )
    loss_eta_part = loss_function(
        reconstructed_particle[:, :, 1][valid_hits_mask],
        particles_data[:, :, 3][valid_hits_mask],
        reduction="sum",
    )
    loss_sin_phi_part = loss_function(
        reconstructed_particle[:, :, 2][valid_hits_mask],
        torch.sin(particles_data[:, :, 2][valid_hits_mask]),
        reduction="sum",
    )
    loss_cos_phi_part = loss_function(
        reconstructed_particle[:, :, 3][valid_hits_mask],
        torch.cos(particles_data[:, :, 2][valid_hits_mask]),
        reduction="sum",
    )
    loss_phi_part = loss_sin_phi_part + loss_cos_phi_part
    loss_pt_part = loss_function(
        reconstructed_particle[:, :, 4][valid_hits_mask],
        particles_data[:, :, 4][valid_hits_mask],
        reduction="sum",
    )

    # Build a dictionary to hold the losses
    rec_loss = {
        "z": loss_z_part,
        "eta": loss_eta_part,
        "phi": loss_phi_part,
        "pt": loss_pt_part,
    }

    return rec_loss


def hit_classification_loss(
    seed_hit_scores: torch.Tensor,
    particles_data: torch.Tensor,
    padded_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the binary cross-entropy loss for hit classification.

    This method classifies hits as seed hits (associated with a particle) or non-seed hits
    (not associated with any particle). It uses binary cross-entropy loss to train the model
    to distinguish between these two classes.

    Args:
        seed_hit_scores: Predicted seed hit scores (probabilities in [0,1]) [batch_size, max_hits]
            (typically taken from the 5th component of the regressed parameters).
        particles_data: Original particle data [batch_size, max_hits, 4]
        padded_mask: Boolean mask for padded hits (True for padded)

    Returns:
        Binary cross-entropy loss scalar tensor
    """
    device = seed_hit_scores.device
    # Check if there are any valid (non-padded) hits
    # padded_mask is 1 for padded hits, so we need to invert it
    non_padded_mask = ~padded_mask.bool()

    if torch.sum(non_padded_mask) == 0:
        # Return zero loss if no valid hits
        return torch.tensor(0.0, device=device)

    # Extract valid (non-padded) data
    valid_scores = seed_hit_scores[non_padded_mask]
    # Inputs are already sigmoid probabilities; clamp for numerical stability
    valid_probs = torch.clamp(valid_scores, min=1e-7, max=1 - 1e-7).squeeze(-1)
    valid_particles = particles_data[non_padded_mask]

    # Create target labels: 1.0 for hits with particles (seed hits), 0.0 for orphan hits
    # pT > 0 indicates valid particle association
    has_particle_mask = valid_particles[:, 4] > 0.0
    target_labels = has_particle_mask.float()

    # Class-balanced weights to mitigate imbalance between orphan (0) and seed (1) hits
    n_total = int(target_labels.numel())
    n_pos = int(torch.sum(target_labels).item())
    n_neg = n_total - n_pos

    if n_pos > 0 and n_neg > 0:
        # Inverse-frequency weights normalized so each class contributes ~ equally
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)
        sample_weights = torch.where(
            target_labels > 0.5,
            torch.full_like(target_labels, w_pos),
            torch.full_like(target_labels, w_neg),
        )
        hit_bce_loss = F.binary_cross_entropy(valid_probs, target_labels, weight=sample_weights, reduction="sum")
    else:
        # Fallback to unweighted if a class is absent to avoid instability
        hit_bce_loss = F.binary_cross_entropy(valid_probs, target_labels, reduction="sum")

    return hit_bce_loss
