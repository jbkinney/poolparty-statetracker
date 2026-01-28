"""Demo: Column configuration for generate_library()"""
import poolparty as pp
import tempfile
import os

# Initialize poolparty
pp.init()

# 1. Without configuration - all columns visible
print("=== Without configuration (all columns visible) ===")
pool1 = pp.from_seqs(['ACGT', 'TGCA'], seq_names=['s1', 's2']).mutagenize(num_mutations=1)
df1 = pool1.generate_library(num_seqs=2, report_design_cards=True)
print("Columns:", df1.columns.tolist())
print()

# 2. With configuration - filtered columns
config_content = """
[columns]
name = true
seq = true
pool_seqs = false     # Hide pool-specific seq columns
pool_states = false   # Hide pool state columns
op_states = false     # Hide operation state columns

[design_cards.from_seqs]
seq_name = true       # Show seq_name
seq_index = false     # Hide seq_index

[design_cards.mutagenize]
positions = true      # Show positions
wt_chars = false      # Hide wt_chars
mut_chars = false     # Hide mut_chars
"""

# Write config to temp file and load it
with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
    f.write(config_content)
    temp_path = f.name

pp.load_config(temp_path)

print("=== With configuration (filtered columns) ===")
pool2 = pp.from_seqs(['ACGT', 'TGCA'], seq_names=['s1', 's2']).mutagenize(num_mutations=1)
df2 = pool2.generate_library(num_seqs=2, report_design_cards=True)
print("Columns:", df2.columns.tolist())
print()
print(df2)
print()

# Clean up
os.unlink(temp_path)

print("=== Configuration successfully controls column visibility! ===")
