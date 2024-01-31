Installing required libraries:
pip install torch imageio numpy cProfile snakeviz

To run:
python -m cProfile -o profiling.prof style-transfer.py

To visualize profiler output:
snakeviz .\profiling.prof