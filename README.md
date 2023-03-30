# UCF ARCC SLURM Multi-Node pytorch training demo

This repo contains the scaffolding for a multi-node pytorch training demo on the UCF ARCC SLURM cluster. Most of the code from this repo is adapted from this [gist](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904).

The code has been adapted to train on the CIFAR-10 dataset. In addition some changes to the slurm script have been modified to work on the UCF ARCC newton cluster.

## Setup

Create a python virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the demo

You can use the `sbatch` slurm command to run this

```bash
sbatch submit.slurm
```

You can follow the output by running `tail -f <JOB_NAME>-<ID>.out`
where `<JOB_NAME>` is the name of the job you specified in the `submit.slurm` file and `<ID>` is the job ID.

## Data download error

When you first submit the job, you may get an error like this:

```bash
RuntimeError: Dataset not found or corrupted. You can use download=True to download it
```

This is expected to occur the first time you run the job. In this case, you can simply cancel the current job re-submit the job and it should work.
