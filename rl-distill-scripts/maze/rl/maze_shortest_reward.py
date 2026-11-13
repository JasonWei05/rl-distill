"""Binary shortest-path reward for the text-maze RL runs (rl_tis_run.sh with RL_REWARD=shortest).

R = 1 iff the response reaches the goal by a shortest path (TailRL's judge_maze_binary_shortest), on the same maze_17
train / val parquets as the goal-reached runs. The returned outcome fields match TailRL's default maze_17 judge, so the
validation metrics keep their names (val-aux/maze_17/is_shortest/mean@16 is the early-stop metric).
"""

from verl.utils.reward_score.maze import judge_maze_binary_shortest


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "maze_17":
        raise NotImplementedError(f"maze_shortest_reward expects maze_17 data, got {data_source}")
    out = judge_maze_binary_shortest(solution_str=solution_str, ground_truth=ground_truth)
    out.pop("length_ratio")  # None off the goal; the default maze_17 judge has no such field
    return out
