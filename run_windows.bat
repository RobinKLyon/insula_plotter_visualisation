@echo off
call conda activate insula-plot
python scripts\run_plot.py --config config.yaml
pause
