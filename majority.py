# File: verl/workers/reward_manager/majority.py

from collections import defaultdict, Counter
import torch
import numpy as np # For array operations if needed for uids

from verl import DataProto
from verl.data_process.s1k import extract_answer

NO_CONSENSUS_TOKEN = "==NO_CONSENSUS_FOUND=="

class MajorityVoteRewardManager:
    """
    A reward manager that first determines a "pseudo-truth" from k generated
    responses per prompt using a majority vote (or other consensus mechanism)
    and then scores each of the k responses against this pseudo-truth.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", config=None, **kwargs) -> None:
        """
        Args:
            tokenizer: The tokenizer for decoding responses.
            num_examine (int): Number of samples to print for debugging.
            compute_score (callable, optional): A function that takes (data_source, solution_str, pseudo_truth_str, extra_info)
                                               and returns a score or a dict with "score" and other metrics.
                                               This will be used to score each original response against the derived pseudo-truth.
                                               If None, a default scoring (e.g., exact match) might be used or an error raised.
            reward_fn_key (str): Key to get data_source from non_tensor_batch.
            config (OmegaConf, optional): The main configuration object, used to get k (rollout.n)
                                           and any majority vote specific configurations.
            **kwargs: Additional keyword arguments.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.NO_CONSENSUS_TOKEN = NO_CONSENSUS_TOKEN
        
        if compute_score is None:
            # Define a simple default or raise an error if no compute_score is provided for this manager
            print("WARN: No 'compute_score' function provided to MajorityVoteRewardManager. "
                  "You'll need to implement how individual responses are scored against the pseudo-truth.")
            # Example default (you'll likely want to make this configurable or more robust)
            def _default_pseudo_truth_scorer(data_source, solution_str, pseudo_truth_str, extra_info):
                return 1.0 if solution_str == pseudo_truth_str else 0.0
            self.compute_score = _default_pseudo_truth_scorer
        else:
            self.compute_score = compute_score
            
        self.reward_fn_key = reward_fn_key
        self.config = config # Store the main config

        # Extract k (number of responses per prompt) from the config
        if self.config is None or not hasattr(self.config, 'actor_rollout_ref') or \
           not hasattr(self.config.actor_rollout_ref, 'rollout') or \
           not hasattr(self.config.actor_rollout_ref.rollout, 'n'):
            raise ValueError("MajorityVoteRewardManager requires 'config.actor_rollout_ref.rollout.n' (k) to be set.")
        self.k_responses_per_prompt = self.config.actor_rollout_ref.rollout.n

        # Get majority vote specific configurations
        self.majority_vote_config = self.config.reward_model.get("majority_vote_specific_config", {})
        self.voting_strategy = self.majority_vote_config.get("strategy", "exact_match_majority") # e.g., "exact_match_majority", "semantic_similarity"
        self.min_votes_ratio_for_truth = self.majority_vote_config.get("min_votes_for_truth_ratio", 0.5) # e.g., > 0.5 for strict majority

        print(f"Initialized MajorityVoteRewardManager with k={self.k_responses_per_prompt}, strategy='{self.voting_strategy}'")


    def _determine_pseudo_truth(self, k_decoded_responses: list[str]) -> str:
        """
        Determines the pseudo-truth from a list of k decoded responses for a single prompt.
        This is where you implement your specific voting/consensus logic.
        """
        processed_k_responses_for_voting = [
            extract_answer(resp_str) for resp_str in k_decoded_responses
        ]
        # print(f"Number of responses: {len(processed_k_responses_for_voting)}")
        # print(f"processed_k_responses_for_voting: {processed_k_responses_for_voting}")
        if self.voting_strategy == "exact_match_majority":
            from collections import Counter # Keep for default
            # if not processed_k_responses_for_voting: return self.NO_CONSENSUS_TOKEN
            processed_k_responses_for_voting = [res for res in processed_k_responses_for_voting if res]
            # print(f"processed_k_responses_for_voting_after_filtering: {processed_k_responses_for_voting}")
            if len(processed_k_responses_for_voting) == 0: return self.NO_CONSENSUS_TOKEN
            vote_counts = Counter(processed_k_responses_for_voting)
            most_common = vote_counts.most_common(1)
            if most_common:
                # You might want the pseudo-truth to be in its "extracted/canonical" form
                # or map it back to one of the original k_decoded_responses if needed later.
                # For now, let's assume the pseudo-truth is also in the processed form.
                min_ratio = self.majority_vote_config.get("min_votes_ratio_for_truth_ratio", 0.0)
                candidate, count = most_common[0]
                if count / len(processed_k_responses_for_voting) >= min_ratio:
                    return candidate
                else:
                    return processed_k_responses_for_voting[0] # Fallback
            return processed_k_responses_for_voting[0]

    def __call__(self, data: DataProto, return_dict=False):
        """
        Computes rewards based on majority vote.
        Assumes `data` contains `k` responses for each original prompt,
        and that these are interleaved if `batch.repeat` was used in RayPPOTrainer.
        """
        if "rm_scores" in data.batch: # If pre-computed RM scores exist, maybe bypass? Or integrate?
            print("WARN: 'rm_scores' found in batch. MajorityVoteRewardManager might ignore them "
                  "or you need to define how they integrate with majority voting.")
            # For now, we'll proceed assuming we always re-calculate based on majority.
            # If you want to use rm_scores, this logic needs to change.

        final_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        final_true_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        reward_extra_info = defaultdict(list)
        
        num_total_responses = len(data) # This is num_original_prompts * k
        # print(f"Data: {data.batch['responses'].shape}, {len(data)}")
        if num_total_responses == 0:
            if return_dict:
                return {"reward_tensor": final_reward_tensor, "true_reward_tensor": final_true_reward_tensor, "reward_extra_info": reward_extra_info}
            return final_reward_tensor

        num_original_prompts = num_total_responses // self.k_responses_per_prompt
        if num_total_responses % self.k_responses_per_prompt != 0:
            raise ValueError(f"Total number of responses ({num_total_responses}) is not divisible by "
                             f"k_responses_per_prompt ({self.k_responses_per_prompt}). Check data preparation.")
        
        # Decode all responses once
        # Assuming 'responses' tensor has shape (num_original_prompts * k, seq_len)
        all_decoded_responses = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                                 for ids in data.batch["responses"]]

        # For printing/debugging
        already_print_data_sources = {}
        num_printed_for_source = defaultdict(int)

        prompts_with_no_consensus_count_this_batch = 0

        # Iterate through each original prompt
        for i_orig_prompt in range(num_original_prompts):
            start_idx = i_orig_prompt * self.k_responses_per_prompt
            end_idx = (i_orig_prompt + 1) * self.k_responses_per_prompt

            # Get the k responses for the current original prompt
            current_k_decoded_responses = all_decoded_responses[start_idx:end_idx]

            # Determine the pseudo-truth for this set of k responses
            pseudo_truth_str = self._determine_pseudo_truth(current_k_decoded_responses)
            # reward_extra_info["pseudo_truth_determined"].append(pseudo_truth_str) # Log for each original prompt

            # print(f"pseudo truth str: {pseudo_truth_str}")

            # current_prompt_had_consensus = (pseudo_truth_str != self.NO_CONSENSUS_TOKEN)
            # if not current_prompt_had_consensus:
            if pseudo_truth_str == self.NO_CONSENSUS_TOKEN:
                prompts_with_no_consensus_count_this_batch += 1

            # Now, score each of the k original responses against this pseudo-truth
            for j_k_response in range(self.k_responses_per_prompt):
                global_idx = start_idx + j_k_response # Index in the full batch
                
                data_item = data[global_idx] # Get the DataProtoItem for this specific response

                # Extract necessary info for scoring and logging for this specific response
                prompt_ids = data_item.batch["prompts"] # This is the prompt for this specific response
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().item()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

                response_ids_for_this_k = data_item.batch["responses"] # This is the current response being scored
                valid_response_length_for_this_k = data_item.batch["attention_mask"][prompt_length:].sum().item()
                
                solution_str_for_this_k = all_decoded_responses[global_idx] # Already decoded

                # Ground truth from the original data (might be unused or used for comparison/logging)
                # If your 'compute_score' doesn't need it, you can remove this.
                original_ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", "N/A_in_majority_vote")
                reward_extra_info["original_ground_truth"].append(original_ground_truth)

                
                data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown_source")
                extra_info_for_scoring = data_item.non_tensor_batch.get("extra_info", None)

                # Use self.compute_score to compare solution_str_for_this_k with pseudo_truth_str
                score_output = self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str_for_this_k,
                    ground_truth=pseudo_truth_str, # Pass pseudo-truth instead of original ground_truth
                    extra_info=extra_info_for_scoring,
                )
                extracted_ans = extract_answer(solution_str_for_this_k)
                print(f"Extracted answer: {extracted_ans}, Psude Ground_truth: {pseudo_truth_str}")
                print(f"Score: {score_output}")

                if isinstance(score_output, dict):
                    reward = score_output["score"]
                    for key, value in score_output.items():
                        reward_extra_info[key].append(value) # Appends per response
                else:
                    reward = float(score_output) # Ensure it's a float
                    reward_extra_info["score"].append(reward) # Store raw score if not a dict

                # Assign reward. GRPO often sums token_level_rewards.
                # If your score is sequence-level, assign it to the last valid token.
                if valid_response_length_for_this_k > 0:
                    final_reward_tensor[global_idx, valid_response_length_for_this_k - 1] = reward
                elif final_reward_tensor.shape[1] > 0: # Handle empty response case, reward on first token
                    final_reward_tensor[global_idx, 0] = reward

                ### Store True Score

                true_score_output = self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str_for_this_k,
                    ground_truth=original_ground_truth,
                    extra_info=extra_info_for_scoring,
                )
                print(f"Extracted answer: {extracted_ans}, Ground_truth: {original_ground_truth}")
                print(f"Score: {true_score_output}")
                reward_extra_info["true_reward"].append(true_score_output)

                if isinstance(true_score_output, dict):
                    true_reward = true_score_output["score"]
                    for key, value in true_score_output.items():
                        reward_extra_info[key].append(value) # Appends per response
                else:
                    true_reward = float(true_score_output) # Ensure it's a float
                    reward_extra_info["true_score"].append(true_reward) # Store raw score if not a dict

                if valid_response_length_for_this_k > 0:
                    final_true_reward_tensor[global_idx, valid_response_length_for_this_k - 1] = true_reward
                elif final_true_reward_tensor.shape[1] > 0: # Handle empty response case, reward on first token
                    final_true_reward_tensor[global_idx, 0] = true_reward

                # Logging for debugging (similar to NaiveRewardManager)
                if data_source not in already_print_data_sources: #This logic might need adjustment
                    already_print_data_sources[data_source] = 0
                
                # if num_printed_for_source[data_source] < self.num_examine:
                #     # Only print once per original prompt for the pseudo-truth, then k times for each scored response
                #     if j_k_response == 0: # Print prompt and pseudo-truth once
                #         print(f"\n--- Examining Original Prompt {i_orig_prompt + 1}/{num_original_prompts} (Data Source: {data_source}) ---")
                #         print("[Prompt]", prompt_str)
                #         print("[Pseudo-Truth Determined]", pseudo_truth_str)
                    
                #     print(f"  --- Sub-Response {j_k_response + 1}/{self.k_responses_per_prompt} ---")
                #     print("  [Original Response]", solution_str_for_this_k)
                #     if isinstance(score_output, dict):
                #         for key, value in score_output.items():
                #             print(f"  [{key} vs Pseudo-Truth]", value)
                #     else:
                #         print("  [Score vs Pseudo-Truth]", reward)
                #     if j_k_response == self.k_responses_per_prompt -1 : # after last sub-response for this prompt
                #         num_printed_for_source[data_source] += 1

        # reward_extra_info["prompts_no_consensus_count"] = prompts_with_no_consensus_count_this_batch
        # print(f"Number of no consensus: {prompts_with_no_consensus_count_this_batch}")
        # print(f"Reward tensor: {final_reward_tensor}")
        # print(stop)
        if return_dict:
            return {
                "reward_tensor": final_reward_tensor,
                "true_reward_tensor": final_true_reward_tensor,
                "reward_extra_info": reward_extra_info, # Contains pseudo_truth_determined (once per orig_prompt)
                                                        # and scores (once per k_response)
            }
        else:
            return final_reward_tensor