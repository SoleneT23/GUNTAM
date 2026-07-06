from typing import Dict
import torch
import torch.nn.functional as F



def attention_loss(
    attention_map_bin: torch.Tensor,                                           
    pairs1: torch.Tensor,                                            
    pairs2: torch.Tensor,                                             
    target: torch.Tensor,                                     
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

                                          
    if pairs1.numel() == 0 or pairs2.numel() == 0:
                                               
        return torch.tensor(0.0, device=attention_map_bin.device)

                                                                        
    logits = attention_map_bin[pairs1, pairs2]

                                     
    targets = (target.float() + 1.0) * 0.5

                                                                                         
    pair_weights = target.abs().float()

    return F.binary_cross_entropy_with_logits(logits, targets, weight=pair_weights, reduction="sum")


def full_attention_loss(
    attention_map_bin: torch.Tensor,                                           
    pairs1: torch.Tensor,                                            
    pairs2: torch.Tensor,                                             
    target: torch.Tensor,                                     
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

                                                                              
    pair_weights_pos = target[pos_mask].abs().float()

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

                                                    
    target_matrix = torch.zeros_like(attention_map_bin, device=device)
    target_matrix[pairs1[pos_mask], pairs2[pos_mask]] = 1.0
    pair_weight_matrix = torch.ones_like(attention_map_bin)
    pair_weight_matrix[pairs1[pos_mask], pairs2[pos_mask]] = pair_weights_pos

    full_mask = torch.ones_like(attention_map_bin, dtype=torch.bool)
    full_mask[num_valid_hits:, :] = False
    full_mask[:, num_valid_hits:] = False

                                                                             
    inactive_cols = torch.ones(attention_map_bin.shape[0], dtype=torch.bool, device=device)
    inactive_cols[pos_hits] = False
    full_mask[:, inactive_cols] = False

    logits = attention_map_bin[full_mask]
    targets = target_matrix[full_mask]

    pos_weight = 1 / max(pairs1[pos_mask].numel(), 1)
    neg_weight = 1 / max(full_mask.sum().item() - pairs1[pos_mask].numel(), 1)

                                                                        
    weights = torch.where(targets > 0, pos_weight * pair_weight_matrix[full_mask], neg_weight)

    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")


def _empty_top_attention_debug(device, dtype, num_real_hits=0, num_unique_real_particle_ids=0):
    empty_scores = torch.empty(0, device=device, dtype=dtype)
    return {
        "positive_scores": empty_scores,
        "negative_scores": empty_scores,
        "random_negative_scores": empty_scores,
        "hard_negative_scores": empty_scores,
        "num_positive_pairs": torch.tensor(0, device=device),
        "num_negative_candidates": torch.tensor(0, device=device),
        "num_selected_negatives": torch.tensor(0, device=device),
        "num_real_hits": torch.tensor(num_real_hits, device=device),
        "num_unique_real_particle_ids": torch.tensor(num_unique_real_particle_ids, device=device),
    }


def _prepare_top_attention_inputs(
    attention_map_bin: torch.Tensor,
    pairs1: torch.Tensor,
    pairs2: torch.Tensor,
    target: torch.Tensor,
    particle_ids: torch.Tensor,
    padding_mask: torch.Tensor,
):
    """Shared preprocessing for the two top-attention losses."""
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
    particle_ids = particle_ids.long()[:seq_len]

    padding_mask = padding_mask.to(device)
    if padding_mask.dim() == 2:
        padding_mask = padding_mask[0]
    else:
        padding_mask = padding_mask.view(-1)
    padding_mask = padding_mask.bool()[:seq_len]

                                                                      
                                                                                    
    real_hit_mask = (~padding_mask) & (particle_ids >= 0)

    real_particle_ids = particle_ids[real_hit_mask]
    unique_real_particle_ids = torch.unique(real_particle_ids)

    pos_mask = target > 0
    if not torch.any(pos_mask):
        return None, _empty_top_attention_debug(
            device=device,
            dtype=dtype,
            num_real_hits=real_hit_mask.sum().item(),
            num_unique_real_particle_ids=unique_real_particle_ids.numel(),
        )

    pos_i_all = pairs1[pos_mask].long().to(device)
    pos_j_all = pairs2[pos_mask].long().to(device)
    pair_weights_all = target[pos_mask].abs().float().to(device)

    in_bounds = (
        (pos_i_all >= 0)
        & (pos_i_all < seq_len)
        & (pos_j_all >= 0)
        & (pos_j_all < seq_len)
    )

    if not torch.any(in_bounds):
        return None, _empty_top_attention_debug(
            device=device,
            dtype=dtype,
            num_real_hits=real_hit_mask.sum().item(),
            num_unique_real_particle_ids=unique_real_particle_ids.numel(),
        )

    pos_i = pos_i_all[in_bounds]
    pos_j = pos_j_all[in_bounds]
    pair_weights_pos = pair_weights_all[in_bounds]

    valid_pos = (
        real_hit_mask[pos_i]
        & real_hit_mask[pos_j]
        & torch.isfinite(attention_map_bin[pos_i, pos_j])
    )

    pos_i = pos_i[valid_pos]
    pos_j = pos_j[valid_pos]
    pair_weights_pos = pair_weights_pos[valid_pos]

    if pos_i.numel() == 0:
        return None, _empty_top_attention_debug(
            device=device,
            dtype=dtype,
            num_real_hits=real_hit_mask.sum().item(),
            num_unique_real_particle_ids=unique_real_particle_ids.numel(),
        )

    prepared = {
        "attention_map_bin": attention_map_bin,
        "device": device,
        "dtype": dtype,
        "seq_len": seq_len,
        "particle_ids": particle_ids,
        "padding_mask": padding_mask,
        "real_hit_mask": real_hit_mask,
        "unique_real_particle_ids": unique_real_particle_ids,
        "pos_i": pos_i,
        "pos_j": pos_j,
        "pair_weights_pos": pair_weights_pos,
        "pos_scores": attention_map_bin[pos_i, pos_j],
    }
    return prepared, None


def _finish_top_attention_loss(
    prepared,
    random_neg_scores: torch.Tensor,
    hard_neg_scores: torch.Tensor,
    num_negative_candidates: torch.Tensor,
    return_debug: bool,
):
    device = prepared["device"]
    dtype = prepared["dtype"]
    pos_scores = prepared["pos_scores"]
    pair_weights_pos = prepared["pair_weights_pos"]
    real_hit_mask = prepared["real_hit_mask"]
    unique_real_particle_ids = prepared["unique_real_particle_ids"]

    mixed_neg_scores = torch.cat([random_neg_scores, hard_neg_scores], dim=0)

    num_pos = pos_scores.numel()
    logits = torch.cat([pos_scores, mixed_neg_scores], dim=0)
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
    weights = torch.cat([pos_weights, neg_weights], dim=0)

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
            "num_negative_candidates": num_negative_candidates,
            "num_selected_negatives": torch.tensor(mixed_neg_scores.numel(), device=device),
            "num_real_hits": torch.tensor(real_hit_mask.sum().item(), device=device),
            "num_unique_real_particle_ids": torch.tensor(unique_real_particle_ids.numel(), device=device),
        }
        return loss, debug_info

    return loss


def top_attention_loss_global_random_hard(
    attention_map_bin: torch.Tensor,
    pairs1: torch.Tensor,
    pairs2: torch.Tensor,
    target: torch.Tensor,
    particle_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    return_debug: bool = False,
    hard_negative_fraction: float = 0.2,
    debug_print: bool = False,
) -> torch.Tensor:
    """
    Top-attention loss using all global different-particle pairs as candidates.

    This is the NxN-mask strategy: it builds the global candidate mask, samples random
    negatives from it, and optionally takes the highest-logit pairs as hard negatives.
    """
    prepared, empty_debug = _prepare_top_attention_inputs(
        attention_map_bin=attention_map_bin,
        pairs1=pairs1,
        pairs2=pairs2,
        target=target,
        particle_ids=particle_ids,
        padding_mask=padding_mask,
    )

    if prepared is None:
        loss = torch.tensor(0.0, device=attention_map_bin.device, dtype=attention_map_bin.dtype)
        if return_debug:
            return loss, empty_debug
        return loss

    attention_map_bin = prepared["attention_map_bin"]
    device = prepared["device"]
    dtype = prepared["dtype"]
    seq_len = prepared["seq_len"]
    particle_ids = prepared["particle_ids"]
    padding_mask = prepared["padding_mask"]
    real_hit_mask = prepared["real_hit_mask"]
    unique_real_particle_ids = prepared["unique_real_particle_ids"]
    pos_i = prepared["pos_i"]
    pos_j = prepared["pos_j"]
    pos_scores = prepared["pos_scores"]
    num_pos = pos_scores.numel()

    empty_scores = torch.empty(0, device=device, dtype=dtype)
    hard_negative_fraction = max(0.0, min(1.0, float(hard_negative_fraction)))

    if unique_real_particle_ids.numel() < 2:
        random_neg_scores = empty_scores
        hard_neg_scores = empty_scores
        num_negative_candidates = torch.tensor(0, device=device)
    else:
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
        num_negative_candidates = torch.tensor(neg_scores.numel(), device=device)
        total_num_neg = min(num_pos, neg_scores.numel())

        if total_num_neg == 0:
            random_neg_scores = empty_scores
            hard_neg_scores = empty_scores
        else:
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
                hard_neg_scores = empty_scores
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
                actual_num_random = min(num_random, remaining_scores.numel())

                if actual_num_random > 0:
                    random_indices = torch.randperm(
                        remaining_scores.numel(),
                        device=device,
                    )[:actual_num_random]
                    random_neg_scores = remaining_scores[random_indices]
                else:
                    random_neg_scores = empty_scores
            else:
                random_neg_scores = empty_scores

    if debug_print:
        pos_left_ids = particle_ids[pos_i]
        pos_right_ids = particle_ids[pos_j]
        pos_same = pos_left_ids == pos_right_ids
        print()
        print("=" * 80)
        print("TOP ATTENTION LOSS DEBUG")
        print("negative_sampling: global")
        print("seq_len:", seq_len)
        print("padding_mask true count:", padding_mask.sum().item())
        print("padding_mask false count:", (~padding_mask).sum().item())
        print("real_hit_mask count:", real_hit_mask.sum().item())
        print("unique_real_particle_ids:", unique_real_particle_ids.numel())
        print("num_pos:", num_pos)
        print("positive same-particle count:", pos_same.sum().item())
        print("positive different-particle count:", (~pos_same).sum().item())
        print("num_negative_candidates:", num_negative_candidates.item())
        print("num_selected_random_negatives:", random_neg_scores.numel())
        print("num_selected_hard_negatives:", hard_neg_scores.numel())
        print("num_selected_negatives:", random_neg_scores.numel() + hard_neg_scores.numel())
        if pos_scores.numel() > 0:
            print("positive sigmoid mean:", torch.sigmoid(pos_scores).mean().item())
        if random_neg_scores.numel() > 0:
            print("random negative sigmoid mean:", torch.sigmoid(random_neg_scores).mean().item())
        if hard_neg_scores.numel() > 0:
            print("hard negative sigmoid mean:", torch.sigmoid(hard_neg_scores).mean().item())
        print("=" * 80)
        print()

    return _finish_top_attention_loss(
        prepared=prepared,
        random_neg_scores=random_neg_scores,
        hard_neg_scores=hard_neg_scores,
        num_negative_candidates=num_negative_candidates,
        return_debug=return_debug,
    )


def top_attention_loss_local_distance(
    attention_map_bin: torch.Tensor,
    pairs1: torch.Tensor,
    pairs2: torch.Tensor,
    target: torch.Tensor,
    particle_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    hits: torch.Tensor,
    return_debug: bool = False,
    hard_negative_fraction: float = 0.2,
    local_negative_k: int = 50,
    local_candidate_pool: int = 512,
    local_coord_dims=(0, 1, 2),
    debug_print: bool = False,
) -> torch.Tensor:
    """
    Top-attention loss using the previous distance/local-negative strategy.

    It does not build an NxN candidate mask. For each sampled positive pair
    (anchor_i, reference_j), it samples a candidate pool of different-particle hits,
    keeps the nearest candidates to reference_j in coordinate space, and draws local
    negatives from those. Hard negatives are the highest-logit negatives inside this
    sampled local pool.
    """
    if hits is None:
        raise ValueError("top_attention_loss_local_distance requires hits.")

    prepared, empty_debug = _prepare_top_attention_inputs(
        attention_map_bin=attention_map_bin,
        pairs1=pairs1,
        pairs2=pairs2,
        target=target,
        particle_ids=particle_ids,
        padding_mask=padding_mask,
    )

    if prepared is None:
        loss = torch.tensor(0.0, device=attention_map_bin.device, dtype=attention_map_bin.dtype)
        if return_debug:
            return loss, empty_debug
        return loss

    attention_map_bin = prepared["attention_map_bin"]
    device = prepared["device"]
    dtype = prepared["dtype"]
    seq_len = prepared["seq_len"]
    particle_ids = prepared["particle_ids"]
    padding_mask = prepared["padding_mask"]
    real_hit_mask = prepared["real_hit_mask"]
    unique_real_particle_ids = prepared["unique_real_particle_ids"]
    pos_i = prepared["pos_i"]
    pos_j = prepared["pos_j"]
    pos_scores = prepared["pos_scores"]
    num_pos = pos_scores.numel()

    empty_scores = torch.empty(0, device=device, dtype=dtype)
    hard_negative_fraction = max(0.0, min(1.0, float(hard_negative_fraction)))

    hits = hits.to(device)
    if hits.dim() == 3:
        hits = hits[0]
    hits = hits[:seq_len]
    coords = hits[:, list(local_coord_dims)].float()
    valid_indices = torch.where(real_hit_mask)[0]

    def sample_local_negative_scores(num_samples: int) -> torch.Tensor:
        if num_samples <= 0 or valid_indices.numel() < 2:
            return empty_scores

        sample_count = int(num_samples)
        candidate_pool = max(1, int(local_candidate_pool))
        nearest_k = max(1, min(int(local_negative_k), candidate_pool))

        pos_choices = torch.randint(
            low=0,
            high=num_pos,
            size=(sample_count,),
            device=device,
        )

        anchor_i = pos_i[pos_choices]
        reference_j = pos_j[pos_choices]
        anchor_particle = particle_ids[anchor_i]

        candidate_positions = torch.randint(
            low=0,
            high=valid_indices.numel(),
            size=(sample_count, candidate_pool),
            device=device,
        )
        candidate_indices = valid_indices[candidate_positions]

        same_particle = particle_ids[candidate_indices] == anchor_particle[:, None]
        for _ in range(8):
            if not same_particle.any():
                break
            replacement_positions = torch.randint(
                low=0,
                high=valid_indices.numel(),
                size=(same_particle.sum().item(),),
                device=device,
            )
            candidate_indices[same_particle] = valid_indices[replacement_positions]
            same_particle = particle_ids[candidate_indices] == anchor_particle[:, None]

        candidate_scores = attention_map_bin[anchor_i[:, None], candidate_indices]
        valid_candidate = (
            (particle_ids[candidate_indices] != anchor_particle[:, None])
            & torch.isfinite(candidate_scores)
        )

        if not valid_candidate.any():
            return empty_scores

        distances = torch.sum(
            (coords[candidate_indices] - coords[reference_j][:, None, :]) ** 2,
            dim=-1,
        )
        distances = torch.where(
            valid_candidate,
            distances,
            torch.full_like(distances, float("inf")),
        )

        closest_positions = torch.topk(
            distances,
            k=nearest_k,
            largest=False,
            sorted=False,
        ).indices

        closest_distances = distances.gather(1, closest_positions)
        valid_closest = torch.isfinite(closest_distances)

        if not valid_closest.any():
            return empty_scores

        random_choice = torch.rand((sample_count, nearest_k), device=device)
        random_choice = torch.where(
            valid_closest,
            random_choice,
            torch.full_like(random_choice, -1.0),
        )

        chosen_in_top = random_choice.argmax(dim=1)
        chosen_positions = closest_positions.gather(
            1,
            chosen_in_top[:, None],
        ).squeeze(1)

        rows = torch.arange(sample_count, device=device)
        neg_j = candidate_indices[rows, chosen_positions]
        keep = (
            valid_closest.any(dim=1)
            & (particle_ids[neg_j] != anchor_particle)
            & torch.isfinite(attention_map_bin[anchor_i, neg_j])
        )

        if not keep.any():
            return empty_scores

        return attention_map_bin[anchor_i[keep], neg_j[keep]]

    if unique_real_particle_ids.numel() < 2:
        random_neg_scores = empty_scores
        hard_neg_scores = empty_scores
        num_negative_candidates = torch.tensor(0, device=device)
    else:
        total_num_neg = num_pos
        num_hard = int(total_num_neg * hard_negative_fraction)
        num_random = total_num_neg - num_hard

        if num_hard > 0:
            local_score_pool_size = max(total_num_neg, num_random + 4 * num_hard)
            local_score_pool_size = min(local_score_pool_size, max(total_num_neg, 8192))
            local_neg_scores = sample_local_negative_scores(local_score_pool_size)
            num_negative_candidates = torch.tensor(local_neg_scores.numel(), device=device)

            if local_neg_scores.numel() == 0:
                hard_neg_scores = empty_scores
                random_neg_scores = empty_scores
            else:
                actual_num_hard = min(num_hard, local_neg_scores.numel())
                hard_neg_scores, hard_indices = torch.topk(
                    local_neg_scores,
                    k=actual_num_hard,
                    largest=True,
                    sorted=False,
                )

                remaining_mask = torch.ones(
                    local_neg_scores.numel(),
                    dtype=torch.bool,
                    device=device,
                )
                remaining_mask[hard_indices] = False
                remaining_scores = local_neg_scores[remaining_mask]

                actual_num_random = min(num_random, remaining_scores.numel())
                if actual_num_random == 0:
                    random_neg_scores = empty_scores
                else:
                    random_indices = torch.randperm(
                        remaining_scores.numel(),
                        device=device,
                    )[:actual_num_random]
                    random_neg_scores = remaining_scores[random_indices]
        else:
            hard_neg_scores = empty_scores
            random_neg_scores = sample_local_negative_scores(num_random)
            num_negative_candidates = torch.tensor(random_neg_scores.numel(), device=device)

    if debug_print:
        pos_left_ids = particle_ids[pos_i]
        pos_right_ids = particle_ids[pos_j]
        pos_same = pos_left_ids == pos_right_ids
        print()
        print("=" * 80)
        print("TOP ATTENTION LOSS DEBUG")
        print("negative_sampling: local")
        print("seq_len:", seq_len)
        print("padding_mask true count:", padding_mask.sum().item())
        print("padding_mask false count:", (~padding_mask).sum().item())
        print("real_hit_mask count:", real_hit_mask.sum().item())
        print("unique_real_particle_ids:", unique_real_particle_ids.numel())
        print("num_pos:", num_pos)
        print("positive same-particle count:", pos_same.sum().item())
        print("positive different-particle count:", (~pos_same).sum().item())
        print("num_negative_candidates:", num_negative_candidates.item())
        print("num_selected_random_negatives:", random_neg_scores.numel())
        print("num_selected_hard_negatives:", hard_neg_scores.numel())
        print("num_selected_negatives:", random_neg_scores.numel() + hard_neg_scores.numel())
        if pos_scores.numel() > 0:
            print("positive sigmoid mean:", torch.sigmoid(pos_scores).mean().item())
        if random_neg_scores.numel() > 0:
            print("random negative sigmoid mean:", torch.sigmoid(random_neg_scores).mean().item())
        if hard_neg_scores.numel() > 0:
            print("hard negative sigmoid mean:", torch.sigmoid(hard_neg_scores).mean().item())
        print("=" * 80)
        print()

    return _finish_top_attention_loss(
        prepared=prepared,
        random_neg_scores=random_neg_scores,
        hard_neg_scores=hard_neg_scores,
        num_negative_candidates=num_negative_candidates,
        return_debug=return_debug,
    )


def top_attention_loss(
    attention_map_bin: torch.Tensor,
    pairs1: torch.Tensor,
    pairs2: torch.Tensor,
    target: torch.Tensor,
    particle_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    hits: torch.Tensor | None = None,
    return_debug: bool = False,
    hard_negative_fraction: float = 0.2,
    negative_sampling: str = "global",
    local_negative_k: int = 50,
    local_candidate_pool: int = 512,
    local_coord_dims=(0, 1, 2),
    debug_print: bool = False,
) -> torch.Tensor:
    """
    Backward-compatible dispatcher.

    negative_sampling="global" -> NxN random/hard-negative loss.
    negative_sampling="local"  -> previous distance-based local-negative loss.
    """
    if negative_sampling == "global":
        return top_attention_loss_global_random_hard(
            attention_map_bin=attention_map_bin,
            pairs1=pairs1,
            pairs2=pairs2,
            target=target,
            particle_ids=particle_ids,
            padding_mask=padding_mask,
            return_debug=return_debug,
            hard_negative_fraction=hard_negative_fraction,
            debug_print=debug_print,
        )

    if negative_sampling == "local":
        return top_attention_loss_local_distance(
            attention_map_bin=attention_map_bin,
            pairs1=pairs1,
            pairs2=pairs2,
            target=target,
            particle_ids=particle_ids,
            padding_mask=padding_mask,
            hits=hits,
            return_debug=return_debug,
            hard_negative_fraction=hard_negative_fraction,
            local_negative_k=local_negative_k,
            local_candidate_pool=local_candidate_pool,
            local_coord_dims=local_coord_dims,
            debug_print=debug_print,
        )

    raise ValueError(
        f"Unknown negative_sampling={negative_sampling}. Use 'global' or 'local'."
    )


def attention_next_loss(
    attention_map_bin: torch.Tensor,                                           
    pairs1: torch.Tensor,                                            
    pairs2: torch.Tensor,                                             
    target: torch.Tensor,                                     
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

                                           
    sources = pairs1[pos_mask]
    targets = pairs2[pos_mask]
    pair_weights = target[pos_mask].float()                                

                                     
    unique_sources = torch.unique(sources)

                                                                               
    attention_logits = attention_map_bin[:num_valid_hits, :num_valid_hits]
    masked_logits = attention_logits[unique_sources].clone()                       

    selected_targets = torch.full_like(unique_sources, fill_value=num_valid_hits)
    source_weights = torch.ones(unique_sources.numel(), device=device)

    if unique_sources.numel() > 0 and sources.numel() > 0:
                                                               
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)          
        targets_row = targets.view(1, -1)

                                                                               
        fwd_mask = source_eq & (targets > sources).view(1, -1)          
        back_mask = source_eq & (targets <= sources).view(1, -1)          

        fwd_min, _ = torch.min(torch.where(fwd_mask, targets_row, torch.full_like(targets_row, num_valid_hits)), dim=1)
        back_max, _ = torch.max(torch.where(back_mask, targets_row, torch.full_like(targets_row, -1)), dim=1)
        fwd_exists = fwd_mask.any(dim=1)

        selected_targets = torch.where(fwd_exists, fwd_min, back_max)

                                                                             
                                                                         
        other_traj_mask = source_eq & (targets_row != selected_targets.view(-1, 1))          
        s_indices, m_indices = torch.where(other_traj_mask)
        if s_indices.numel() > 0:
            other_traj_cols = torch.zeros(unique_sources.numel(), num_valid_hits, dtype=torch.bool, device=device)
            other_traj_cols[s_indices, targets[m_indices]] = True
            masked_logits[other_traj_cols] = float("-inf")

                                  
        weights_row = pair_weights.view(1, -1).expand(unique_sources.numel(), -1)          
        source_weights = torch.max(torch.where(source_eq, weights_row, torch.zeros_like(weights_row)), dim=1).values       

    per_sample_loss = F.cross_entropy(masked_logits, selected_targets, reduction="none")
    loss = (per_sample_loss * source_weights).sum()

    return loss


def attention_backward_loss(
    attention_map_bin: torch.Tensor,                                           
    pairs1: torch.Tensor,                                            
    pairs2: torch.Tensor,                                             
    target: torch.Tensor,                                     
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

                                           
    sources = pairs1[pos_mask]
    targets = pairs2[pos_mask]
    pair_weights = target[pos_mask].abs().float()                                

                                     
    unique_sources = torch.unique(sources)

                               
                                                
                                                                   
    selected_targets = torch.full_like(unique_sources, fill_value=num_valid_hits)

                                                                                               
    if unique_sources.numel() > 0 and sources.numel() > 0:
                                                               
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)          

                                                                                 
        forward_pairs_mask = targets >= sources       
        backward_pairs_mask = targets < sources       

                             
        fwd_mask = source_eq & forward_pairs_mask.view(1, -1)
        back_mask = source_eq & backward_pairs_mask.view(1, -1)

        targets_row = targets.view(1, -1)

                                                                      
        back_candidates = torch.where(back_mask, targets_row, torch.full_like(targets_row, -1))
        back_max, _ = torch.max(back_candidates, dim=1)       
        back_exists = back_mask.any(dim=1)       

                                                                                 
        fwd_candidates = torch.where(fwd_mask, targets_row, torch.full_like(targets_row, num_valid_hits))
        fwd_min, _ = torch.min(fwd_candidates, dim=1)       

                                                 
        selected_targets = torch.where(back_exists, back_max, fwd_min)
                                             

                                                                               
    attention_logits = attention_map_bin[:num_valid_hits, :num_valid_hits]

                                                                                            
    if unique_sources.numel() > 0 and sources.numel() > 0:
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)          
        weights_row = pair_weights.view(1, -1).expand(unique_sources.numel(), -1)          
        source_weights = torch.max(torch.where(source_eq, weights_row, torch.zeros_like(weights_row)), dim=1).values       
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

                                                    
                                                               
    non_padded_mask = ~padded_mask.bool()

                                                                                   
                                                                                              
    has_particle_mask = particles_data[:, :, 4] > 0.0                                        
    valid_hits_mask = non_padded_mask & has_particle_mask

    if torch.sum(valid_hits_mask) == 0:
                                                            
        return {
            "z": torch.tensor(0.0, device=device),
            "eta": torch.tensor(0.0, device=device),
            "phi": torch.tensor(0.0, device=device),
            "pt": torch.tensor(0.0, device=device),
        }
                                                    

                              
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
                                                    
                                                               
    non_padded_mask = ~padded_mask.bool()

    if torch.sum(non_padded_mask) == 0:
                                           
        return torch.tensor(0.0, device=device)

                                     
    valid_scores = seed_hit_scores[non_padded_mask]
                                                                             
    valid_probs = torch.clamp(valid_scores, min=1e-7, max=1 - 1e-7).squeeze(-1)
    valid_particles = particles_data[non_padded_mask]

                                                                                        
                                                 
    has_particle_mask = valid_particles[:, 4] > 0.0
    target_labels = has_particle_mask.float()

                                                                                       
    n_total = int(target_labels.numel())
    n_pos = int(torch.sum(target_labels).item())
    n_neg = n_total - n_pos

    if n_pos > 0 and n_neg > 0:
                                                                                  
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)
        sample_weights = torch.where(
            target_labels > 0.5,
            torch.full_like(target_labels, w_pos),
            torch.full_like(target_labels, w_neg),
        )
        hit_bce_loss = F.binary_cross_entropy(valid_probs, target_labels, weight=sample_weights, reduction="sum")
    else:
                                                                          
        hit_bce_loss = F.binary_cross_entropy(valid_probs, target_labels, reduction="sum")

    return hit_bce_loss
