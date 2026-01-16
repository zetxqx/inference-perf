# Useful commands

* run inference_perf on batch job

```sh
NS=<your_namespace>
helm upgrade -i inference-perf deploy/inference-perf -f deploy/inference-perf/valuesqwen32b-pc.yaml -n $NS
helm uninstall inference-perf -n $NS
```


* (optional) build my own inference_perf image and push

```sh
docker build -t us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/inference-perf:latest .
docker push us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/inference-perf:latest
```
