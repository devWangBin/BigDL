# Nano Installation

Note: For windows users, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Nano. Please refer to [Nano Windows install guide](../Howto/windows_guide.md) for instructions.


BigDL-Nano can be installed using pip and we recommend installing BigDL-Nano in a conda environment.

For PyTorch Users, you can install bigdl-nano along with some dependencies specific to PyTorch using the following commands.

```bash
conda create -n env
conda activate env
pip install --pre --upgrade bigdl-nano[pytorch]
```

For TensorFlow users, you can install bigdl-nano along with some dependencies specific to TensorFlow using the following commands.

```bash
conda create -n env
conda activate env
pip install --pre --upgrade bigdl-nano[tensorflow]
```

We also partially support M-series chip users with no guarantee of acceleration with same API. Currently only tensorflow is experimentally supported.

```bash
conda create -n env python=3.8
conda activate env
conda install -c apple tensorflow-deps
pip install --pre --upgrade bigdl-nano[tensorflow]
```

```eval_rst
.. note::
    Since bigdl-nano is still in the process of rapid iteration, we highly recommend that you install nightly build version through the above command to facilitate your use of the latest features.

    For stable version, please refer to the document and installation guide `here <https://bigdl.readthedocs.io/en/v2.1.0/doc/Nano/Overview/nano.html>`_ .
```

After installing bigdl-nano, you can run the following command to setup a few environment variables.

```bash
source bigdl-nano-init
```

The `bigdl-nano-init` scripts will export a few environment variable according to your hardware to maximize performance.

In a conda environment, `source bigdl-nano-init` will also be added to `$CONDA_PREFIX/etc/conda/activate.d/`, which will automaticly run when you activate your current environment.

In a pure pip environment, you need to run `source bigdl-nano-init` every time you open a new shell to get optimal performance and run `source bigdl-nano-unset-env` if you want to unset these environment variables.

---