# Kubernetes

Running the complete DeepEcho Benchmarking suite can take a long time.

For this reason, it comes prepared to be executed distributedly over a dask cluster created using Kubernetes.

## Table of Contents

* [Requirements](#requirements)
    * [Kubernetes Cluster](#kubernetes-cluster)
    * [Admin Acces](#admin-access)
    * [Dask Function](#dask-function)
* [Benchmark Configuration](#benchmark-configuration)
    * [Configuration Format](#configuration-format)
    * [Configuration Examples](#configuration-examples)
* [Run a function on Kubernetes](#run-a-function-on-kubernetes)
    * [Usage Example](#usage-example)

## Requirements

### Kubernetes Cluster

The current version of the DeepEcho benchmark is only prepared to run on kubernetes clusters for which direct access is enabled from the system that is triggering the commands, such as self-hosted clusters or AWS EKS clusters created using `eksctl`.

You can easily make sure of this by running the following command:

```bash
kubectl --version
```

If the output does not show any errors, you should be good to go!

### Admin Access

For the current version, you need to execute the DeepEcho benchmark from a POD inside the cluster within a namespace for which admin access is granted.

If you are running your benchmark POD inside the default workspace, you can create the necessary roles using the following yml config:

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

NOTE: A yml file named `dask-admin.yml` with this configuration can be found inside the `benchmark/kubernetes` folder.

Once you have downloaded or created this file, run `kubectl apply -f dask-admin.yml`.

### Dask Function

The Kubernetes framework allows running any function on a distributed cluster, as far as it uses Dask to distribute its tasks and its output is a `pandas.DataFrame`.

In particular, the `run_benchmark` function from the DeepEcho Benchmarking framework already does it, so all you need to do is execute it.

## Benchmark Configuration

In order to run a dask function on Kubernetes you will need to create a
config dictionary to indicate how to setup the cluster and what to run in it.

### Configuration Format

The config dict that needs to be provided to the `run_dask_function` function has the following entries:

* `run`: specification of what function needs to be run and its arguments.
* `dask_cluster`: configuration to use when creating the dask cluster.
* `output`: where to store the output from the executed function.

#### run

Within the `run` section you need to specify:

* `function`: The complete python path to the function to be run. When runing `DeepEcho`, this value must be set to `deepecho.benchmark.run_benchmark`.
* `args`: A dictionary containing the keyword args that will be used with the given function.

#### dask_cluster

Within the `dask_cluster` section you can specify:
* `workers`: The amount of workers to use. This can be specified as a fixed integer value or as a subdictionary specifying a range, so dask-kubernetes can adapt the cluster size to the work load:
    * `minimum`: minumum number of dask workers to create.
    * `maximum`: maximum number of dask workers to create.
* `worker_config`: specification about how to setup each worker

##### worker_config

* `resources`: A dictionary containig the following keys:
    * `memory`: The amount of RAM memory.
    * `cpu`: The amount of cpu's to use.
* `image`: A docker image to be used. If not specified, you must specify the `git_repository`.
* `setup`: (Optional) spectification of any additional things to install or run to initialize the container before starting the dask-cluster.

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
* `secret_key`: AWS secrect authentication key to access the bucket.

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
        'worker_config': {
            'resources': {
                'memory': '4G',
                'cpu': 4
            },
            'image': 'mydockerimage:latest',
        },
    },
    'output': {
        'path': 'results/my_results.csv',
        'bucket': 'my-s3-bucket',
        'key': 'myawskey',
        'secret_key': 'myawssecretkey'
    }
}
```

## Run a function on Kubernetes.

Create a pod, using the local kubernetes configuration, that starts a Dask Cluster using dask-kubernetes and runs a function specified within the `config` dictionary. Then, this pod talks to kubernetes to create `n` amount of new `pods` with a dask worker inside of each forming a `dask` cluster. Then, a function specified from `config` is being imported and run with the given arguments. The tasks created by this `function` are being run on the `dask` cluster for distributed computation.

Arguments:

* `config`: config dictionary.
* `namespace`: namespace where the dask cluster will be created.

### Usage Example

In this usage example we will create a config dictionary that will run the `deepecho.benchmark.run_benchmark` function. For our `dask_cluster` we will be requesting 2 workers and giving them 4 cores / cpu's to each one to work with and the docker image `mlbazaar/btb:latest`. Then we will call `run_on_benchmark` to create the pods and we will see the logs of the pod that created the workers.

1. First write your config dict following the [instructions above](#benchmark-configuration).
2. Once you have your *config* dict you can import the `run_on_kubernetes` function to create the first pod.

```python
from deepecho.benchmark.kubernetes import run_on_kubernetes


run_on_kubernetes(config)
```

3. If everything proceeded as expected, a message `Pod created` should be displayed on your end. Then, you can check the pod's state or logs by runing the following commands in your console:

```bash
kubectl get pods
```

*Note*: bear in mind that this pod will create more pods, with the config that we provided there should be a total of 3 pods (the one that launched the task and the two workers that we specified).

4. Once you have the name of the pod (it's usually the name of the image used with a unique extension and with the lowest runing time if you run the command after your code finished executing) you can run the following command to see its logs:

```bash
kubectl logs -f <name-of-the-pod>
```
