# Kubernetes

Running the complete DeepEcho Benchmarking suite can take a long time.

For this reason, it comes prepared to be executed on a distributed dask cluster created
using Kubernetes.

## Table of Contents

* [Requirements](#requirements)
    * [Kubernetes Cluster](#kubernetes-cluster)
    * [Admin Access](#admin-access)
    * [Dask Function](#dask-function)
* [Benchmark Configuration](#benchmark-configuration)
    * [Configuration Format](#configuration-format)
    * [Configuration Examples](#configuration-examples)
* [Run a function on Kubernetes](#run-a-function-on-kubernetes)
    * [Running on AWS EKS](#running-on-aws-eks)

## Requirements

### Kubernetes Cluster

The current version of the DeepEcho benchmark is only prepared to run on kubernetes clusters
for which direct access is enabled from the system that is triggering the commands, such as
self-hosted clusters or AWS EKS clusters created using `eksctl`.

You can easily make sure of this by running the following command:

```bash
kubectl --version
```

If the output does not show any errors, you should be good to go!

### Admin Access

For the current version, you need to execute the DeepEcho benchmark from a POD inside the
cluster within a namespace for which admin access is granted.

If you are running your benchmark POD inside the default workspace, you can create the
necessary roles using the following yml config:

```yaml=
kind: ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: dask-admin
rules:
- apiGroups:
    - ""
  resources:
    - pods
    - services
  verbs:
    - list
    - create
    - delete
---
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: dask-admin
subjects:
- kind: ServiceAccount
  name: default
  namespace: default
roleRef:
  kind: ClusterRole
  name: dask-admin
  apiGroup: rbac.authorization.k8s.io
```

NOTE: A yml file named `dask-admin.yml` with this configuration can be found inside the
`benchmark/config` folder.

Once you have downloaded or created this file, run `kubectl apply -f dask-admin.yml`.

### Dask Function

The Kubernetes framework allows running any function on a distributed cluster, as far as
it uses Dask to distribute its tasks and its output is a `pandas.DataFrame`.

In particular, the `run_benchmark` function from the DeepEcho Benchmarking framework already
does it, so all you need to do is execute it.

## Benchmark Configuration

In order to run a dask function on Kubernetes you will need to create a
config dictionary to indicate how to setup the cluster and what to run in it.

### Configuration Format

The config dict that needs to be provided to the `run_dask_function` function has the
following entries:

* `run`: specification of what function needs to be run and its arguments.
* `dask_cluster`: configuration to use when creating the dask cluster.
* `output`: where to store the output from the executed function.

#### run

Within the `run` section you need to specify:

* `function`: The complete python path to the function to be run. When running `DeepEcho`,
  this value must be set to `deepecho.benchmark.run_benchmark`.
* `args`: A dictionary containing the keyword args that will be used with the given function.

#### dask_cluster

Within the `dask_cluster` section you can specify:
* `workers`: The amount of workers to use. This can be specified as a fixed integer value or as
  a subdictionary specifying a range, so dask-kubernetes can adapt the cluster size to the work
  load:
    * `minimum`: minimum number of dask workers to create.
    * `maximum`: maximum number of dask workers to create.
* `image`: A docker image to be used. If not specified, you must specify the `git_repository`.
* `setup`: (Optional) specification of any additional things to install or run to initialize
  the container before starting the dask-cluster.
* `image`: specification about how to setup each worker
* `worker_resources`: Specification of the resources to be used by the workers, as a dictionary
  containing any combination of the following keys:
    * `memory`: The amount of RAM memory.
    * `cpu`: The amount of cpu's to use.
    * `nvidia.com/gpu`: Number of GPUs to use.
* `master_resources`: Specification of the resources to be used by the master, as a dictionary
  containing any combination of the following keys:
    * `memory`: The amount of RAM memory.
    * `cpu`: The amount of cpu's to use.
    * `nvidia.com/gpu`: Number of GPUs to use.

###### setup

* `script`: Location to bash script from the docker container to be run.
* `git_repository`: A dictionary containing the following keys:
    * `url`: Link to the github repository to be cloned.
    * `reference`: A reference to the branch or commit to checkout at.
    * `install`: command run to install this repository.
* `pip_packages`: A list of pip packages to be installed.
* `apt_packages`: A list of apt packages to be installed.

#### output

* `path`: The path to a local file or s3 were the file will be saved.
* `bucket`: If given, the path specified previously will be saved as s3://bucket/path
* `key`: AWS authentication key to access the bucket.
* `secret_key`: AWS secret authentication key to access the bucket.

### Configuration Examples

Here is an example of such a config dictionary that uses a custom docker image:

```python
config = {
    'run': {
        'function': 'deepecho.benchmark.run_benchmark',
        'args': {
            'datasets': 20,
            'moels': ['PARModel'],
        }
    },
    'dask_cluster': {
        'workers': 4,
        'master_resources': {
            'memory': '4G',
            'cpu': 4,
        },
        'worker_resources': {
            'memory': '4G',
            'cpu': 4,
            'nvidia.com/gpu': 1,
        },
        'image': 'mydockerimage:latest',
    },
    'output': {
        'path': 'results/my_results.csv',
        'bucket': 'my-s3-bucket',
        'key': 'myawskey',
        'secret_key': 'myawssecretkey',
    }
}
```

## Run a function on Kubernetes.

Create a pod, using the local kubernetes configuration, that starts a Dask Cluster using
dask-kubernetes and runs a function specified within the `config` dictionary. Then, this pod
talks to kubernetes to create `n` amount of new `pods` with a dask worker inside of each forming
a `dask` cluster. Then, a function specified from `config` is being imported and run with the
given arguments. The tasks created by this `function` are being run on the `dask` cluster for
distributed computation.

Arguments:

* `config`: config dictionary.
* `namespace`: namespace where the dask cluster will be created.

## Running on AWS EKS

This is a very short guide of the necessary steps to run the DeepEcho benchmark on a Kubernetes
Cluster deployed on AWS EKS.

### Requirements

Before you execute the following steps you need to ensure that:

* You have `eksctl` installed and ready to use.
* You have your AWS credentials properly configured in `~/.aws/credentials`.
* The `default` AWS credentials are the ones that you will be using (`eksctl` has a bug that does not
  allow the creation of GPU-enabled clusters when using `--profile=XXX`).
* Your IAM user has all the required permissions to operate on EKS and ECS.

### Create and configure a cluster

#### Create a cluster

In order to create a cluster you need to execute the `eksctl create cluster` command passing
the type and number of worker nodes to create.

Notice that if you want to run using GPU (recommended), you will need to create instances
of types `P3`, `P2`, `G4` or `G3`.

For example, in order to create a cluster with 2 nodes with 1 GPU each, you can run:

```bash
eksctl create cluster --node-type=g4dn.xlarge --nodes=2
```

#### Enable GPUs on the nodes

**NOTE**: You can omit this step if you do not intend to run using GPU (not recommended).

Once you have created the cluster, you will need to enable the GPU usage by installing
the `nvidia-device-plugin` in the cluster with the following command:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.11/nvidia-device-plugin.yml
```

#### Set permissions

Finally, you will need to create a special role in your cluster to allow `dask-kubernetes` to
create and manage pods for you.

For this, you can execute the `dask-admin.yml` file provided within this repository:

```
kubectl apply -f dask-admin.yml
```

### Edit your config

As explained above, in order to run the DeepEcho Benchmark on Kubernetes you will need a config
file.

Inside the `benchmark/config` folder you will find the `benchmark_config.yml` file, which is
the one we use to run the complete benchmark suite.

If you want to customize your execution you can edit it to set:

* Additional arguments to for the `run_benchmark` function, within the section `run.args`, such
  as the datasets to use, the models to benchmark, the maximum number of entities per dataset
  or the model hyperparameters.
* Different number of workers or resources. Remember to remove the `nvidia.com/gpu` entry
  if you do not have GPUs available or do not want to use them.

If you want to upload the results of the execution to S3 after the benchmark is finished,
also update the `output` section accordingly.

### Run the benchmark

Once you have created the Kubernetes cluster and edited the configuration, you can start
the benchmark by using the following command:

```bash
python -m deepecho.benchmark.kubernetes path/to/your/benchmark_config.yml --create-pod
```

This will start a POD that will orchestrate the execution of the benchmark, creating as many
worker PODS as indicated in the configuration, and at the end upload the results to you S3 bucket.

**NOTE**: Do not forget to delete the POD and the EKS Cluster once the execution has finished!
