from flax.traverse_util import flatten_dict


def summarize_params(params):
    flat = flatten_dict(params, sep="/")
    total = 0
    for k, v in flat.items():
        count = v.size
        total += count
        print(f"{k:60} {v.shape} {v.dtype} ({count:,} params)")
    print(f"\nTotal parameters: {total:,}")

