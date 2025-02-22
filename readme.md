## create a conda env using this:
conda create --name final_env

## activate the env using this:
conda activate final_env

conda install -c conda-forge detectron2

cd PaddleOCR
python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

