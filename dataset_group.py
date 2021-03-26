from torch.utils.data import Dataset

import torch
import config
import numpy as np

from scipy.stats import norm

from tqdm import tqdm


class DKTDataset(Dataset):
    def __init__(self, group, max_seq, min_seq, overlap_seq, user_performance, n_levels, mu_itv):
        self.samples = group
        self.max_seq = max_seq
        self.min_seq = min_seq
        self.overlap_seq = overlap_seq
        self.user_performance = user_performance
        self.data = []
        self.n_levels = n_levels
        self.mu_itv = mu_itv
        self.mu_levels, self.std_levels = self._fit_norm(user_performance)
        for user_id, (exercise, part, correctness, elapsed_time, lag_time_s, lag_time_m, lag_time_d, p_explanation) in tqdm(self.samples.items(), total=len(self.samples), desc="Loading Dataset"):
            content_len = len(exercise)
            if content_len < self.min_seq:
                continue  # skip sequence with too few contents

            if content_len > self.max_seq:
                initial = content_len % self.max_seq
                if initial >= self.min_seq:
                    self.data.extend([(user_id, np.append([config.START], exercise[:initial]),
                                       np.append([config.START], part[:initial]),
                                       np.append([config.START], correctness[:initial]),
                                       np.append([config.START], elapsed_time[:initial]),
                                       np.append([config.START], lag_time_s[:initial]),
                                       np.append([config.START], lag_time_m[:initial]),
                                       np.append([config.START], lag_time_d[:initial]),
                                       np.append([config.START], p_explanation[:initial]))])
                for seq in range(content_len // self.max_seq):
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.data.extend([(user_id, np.append([config.START], exercise[start: end]),
                                       np.append([config.START], part[start: end]),
                                       np.append([config.START], correctness[start: end]),
                                       np.append([config.START], elapsed_time[start: end]),
                                       np.append([config.START], lag_time_s[start: end]),
                                       np.append([config.START], lag_time_m[start: end]),
                                       np.append([config.START], lag_time_d[start: end]),
                                       np.append([config.START], p_explanation[start: end]))])
            else:
                self.data.extend([(user_id, np.append([config.START], exercise),
                                   np.append([config.START], part),
                                   np.append([config.START], correctness),
                                   np.append([config.START], elapsed_time),
                                   np.append([config.START], lag_time_s),
                                   np.append([config.START], lag_time_m),
                                   np.append([config.START], lag_time_d),
                                   np.append([config.START], p_explanation))])

    def _fit_norm(self, user_perf):
        data = [d for d in user_perf.values()]
        mu, std = norm.fit(data)
        mu_levels = [mu - (self.n_levels - 1) * self.mu_itv / 2 + i * self.mu_itv for i in range(self.n_levels)]
        std_levels = [np.sqrt(std ** 2 / self.n_levels) for _ in range(self.n_levels)]
        return mu_levels, std_levels

    def _predict_level(self, user_perf, mu_levels, std_levels):
        probs = []
        for mu, std in zip(mu_levels, std_levels):
            probs.append(norm.pdf(user_perf, mu, std))
        probs = np.array(probs)
        probs = probs / sum(probs)
        return probs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_user_id, raw_content_ids, raw_part, raw_correctness, raw_elapsed_time, raw_lag_time_s, raw_lag_time_m, raw_lag_time_d, raw_p_explan = self.data[idx]
        if raw_user_id in self.user_performance:
            user_per = self.user_performance[raw_user_id]
            probs = self._predict_level(user_per, self.mu_levels, self.std_levels)
        else:
            probs = np.ones(len(self.mu_levels))
            probs /= len(self.mu_levels)
        seq_len = len(raw_content_ids)

        input_content_ids = np.zeros(self.max_seq, dtype=np.int64)
        input_part = np.zeros(self.max_seq, dtype=np.int64)
        input_correctness = np.zeros(self.max_seq, dtype=np.int64)
        input_elapsed_time = np.zeros(self.max_seq, dtype=np.int64)
        input_lag_time_s = np.zeros(self.max_seq, dtype=np.int64)
        input_lag_time_m = np.zeros(self.max_seq, dtype=np.int64)
        input_lag_time_d = np.zeros(self.max_seq, dtype=np.int64)
        input_p_explan = np.zeros(self.max_seq, dtype=np.int64)

        label = np.zeros(self.max_seq, dtype=np.int64)

        if seq_len == self.max_seq + 1:  # START token
            input_content_ids[:] = raw_content_ids[1:]
            input_part[:] = raw_part[1:]
            input_p_explan[:] = raw_p_explan[1:]
            input_correctness[:] = raw_correctness[:-1]
            input_elapsed_time[:] = np.append(raw_elapsed_time[0], raw_elapsed_time[2:])
            input_lag_time_s[:] = np.append(raw_lag_time_s[0], raw_lag_time_s[2:])
            input_lag_time_m[:] = np.append(raw_lag_time_m[0], raw_lag_time_m[2:])
            input_lag_time_d[:] = np.append(raw_lag_time_d[0], raw_lag_time_d[2:])
            label[:] = raw_correctness[1:] - 2
        else:
            input_content_ids[-(seq_len - 1):] = raw_content_ids[1:]  # Delete START token
            input_part[-(seq_len - 1):] = raw_part[1:]
            input_p_explan[-(seq_len - 1):] = raw_p_explan[1:]
            input_correctness[-(seq_len - 1):] = raw_correctness[:-1]
            input_elapsed_time[-(seq_len - 1):] = np.append(raw_elapsed_time[0], raw_elapsed_time[2:])
            input_lag_time_s[-(seq_len - 1):] = np.append(raw_lag_time_s[0], raw_lag_time_s[2:])
            input_lag_time_m[-(seq_len - 1):] = np.append(raw_lag_time_m[0], raw_lag_time_m[2:])
            input_lag_time_d[-(seq_len - 1):] = np.append(raw_lag_time_d[0], raw_lag_time_d[2:])
            label[-(seq_len - 1):] = raw_correctness[1:] - 2

        _input = {"content_id": input_content_ids.astype(np.int64),
                  "part": input_part.astype(np.int64),
                  "correctness": input_correctness.astype(np.int64),
                  "elapsed_time": input_elapsed_time.astype(np.int64),
                  "lag_time_s": input_lag_time_s.astype(np.int64),
                  "lag_time_m": input_lag_time_m.astype(np.int64),
                  "lag_time_d": input_lag_time_d.astype(np.int64),
                  "prior_explan": input_p_explan.astype(np.int64)}
        return _input, label, probs
