import numpy as np

scaling_method = "minmax"  # or "standard"

# Read min and max
with open("../data/SS_2026/spcs_grid/EF_minmax.txt", 'r') as f:
    EF_min, EF_max = map(float, f.readline().split())

print("EF_min:", EF_min, "EF_max:", EF_max)

if scaling_method == "minmax":
    # Scale to [0,1]
    def scale(EF):
        return (EF - EF_min) / (EF_max - EF_min)
elif scaling_method == "standard":
    # Standard score (z-score)
    EF_mean = (EF_max + EF_min) / 2
    EF_std = (EF_max - EF_min) / 2
    def scale(EF):
        return (EF - EF_mean) / EF_std
else:
    raise ValueError("Unknown scaling method")

# Example usage
# scaled_EF = scale(EF_array)

