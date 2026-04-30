# # debug_episode.py — drop in the project root
# import sys
# from pathlib import Path

# REPO = Path(__file__).resolve().parent
# sys.path.insert(0, str(REPO))

# from env import Environment

# REFERENCE_SOLUTION = '''\
# import torch

# @torch.no_grad()
# def generate(model, input_ids, max_tokens):
#     logits, past_kvs = model(input_ids, past_kvs=None)
#     next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
#     out = torch.cat([input_ids, next_tok], dim=1)
#     for _ in range(max_tokens - 1):
#         logits, past_kvs = model(next_tok, past_kvs=past_kvs)
#         next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
#         out = torch.cat([out, next_tok], dim=1)
#     return out
# '''

# env = Environment(episode_files=REPO)
# env.reset()

# # Write the solution
# print(">>> writing generate.py")
# r = env.step({"type": "write_file", "path": "generate.py", "content": REFERENCE_SOLUTION})
# print(f"    obs: {r.observation}")

# # Sanity: list files in the workdir from outside
# print(f"\n>>> workdir contents:")
# for f in sorted(env.sandbox.workdir.iterdir()):
#     print(f"    {f.name}")

# # Run judge directly via exec, NOT submit, so we see raw output
# print(f"\n>>> running judge.py via exec")
# r = env.step({"type": "exec", "cmd": ["python", "judge.py"]})
# print(r.observation)

# env.close()