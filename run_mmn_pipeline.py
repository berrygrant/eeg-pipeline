# run_mmn_pipeline.py
#!/usr/bin/env python3
"""
Entry point for the modular MMN pipeline.

Usage:
  python run_mmn_pipeline.py --raw_dir ... --subject_csv_dir ... --out_dir ...

Summary:
  python run_mmn_pipeline.py --raw_dir ... --subject_csv_dir ... --out_dir /tmp \
      --summarize_one_file /path/to/s203.vhdr
"""
from mmn_pipeline.cli import main

if __name__ == "__main__":
    main()