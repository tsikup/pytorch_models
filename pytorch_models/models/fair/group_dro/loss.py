import numpy as np
import torch


class GroupDROLoss(torch.nn.Module):
    def __init__(
        self,
        criterion: torch.nn.Module,
        is_robust: bool,
        n_groups: int,
        group_counts: np.ndarray,
        alpha: float = None,
        gamma: float = 0.1,
        adj: np.ndarray = None,
        min_var_weight: float = 0.0,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        btl: bool = False,
    ):
        super(GroupDROLoss, self).__init__()
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.relocated = False

        self.n_groups = n_groups
        self.group_counts = torch.from_numpy(group_counts)
        self.group_frac = self.group_counts / self.group_counts.sum()
        # self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float()
        else:
            self.adj = torch.zeros(self.n_groups).float()

        if is_robust:
            assert alpha, "alpha must be specified"

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte()

        self.reset_stats()

    def relocate(self, device):
        if not self.relocated:
            self.adj = self.adj.to(device)
            self.group_counts = self.group_counts.to(device)
            self.adv_probs = self.adv_probs.to(device)
            self.exp_avg_loss = self.exp_avg_loss.to(device)
            self.exp_avg_initialized = self.exp_avg_initialized.to(device)
            self.processed_data_counts = self.processed_data_counts.to(device)
            self.update_data_counts = self.update_data_counts.to(device)
            self.update_batch_counts = self.update_batch_counts.to(device)
            self.avg_group_loss = self.avg_group_loss.to(device)

            self.relocated = True

    def forward(self, logits, y, group_idx=None):
        self.relocate(logits.device)
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(
            logits, y.float() if logits.shape[1] == 1 else y.squeeze()
        )
        group_loss, group_count = self.compute_group_avg(
            per_sample_losses, group_idx, logits.device
        )

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (
            1 - self.min_var_weight
        )

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx, device):
        # compute observed counts and mean loss for each group
        group_map = (
            group_idx.squeeze()
            == torch.arange(self.n_groups).unsqueeze(1).long().to(device)
        ).float()  # size: 2 x batch_size
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        # import pdb; pdb.set_trace()

        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups)
        self.update_data_counts = torch.zeros(self.n_groups)
        self.update_batch_counts = torch.zeros(self.n_groups)
        self.avg_group_loss = torch.zeros(self.n_groups)
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.batch_count = 0.0

    def update_stats(self, actual_loss, group_loss, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = (
            prev_weight * self.avg_group_loss + curr_weight * group_loss
        )

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
            1 / denom
        ) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
