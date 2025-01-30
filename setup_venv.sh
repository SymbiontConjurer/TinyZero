if [ "${PWD##*/}" != "TinyZero" ]; then
    echo "Please run this script from the repo root."
    exit 1
fi


if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    exit 0
fi

python -m venv venv
source venv/bin/activate

pip install vllm==0.6.3 ray
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib
