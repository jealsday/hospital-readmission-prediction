#!/usr/bin/env bash
# Deploy the prediction service to a local Kubernetes cluster created with kind.
# Run from the project root: bash deploy/kubernetes/deploy.sh
set -euo pipefail

CLUSTER=readmission
IMAGE=hospital-readmission:v1

docker build -t "$IMAGE" .

if ! kind get clusters | grep -qx "$CLUSTER"; then
    kind create cluster --name "$CLUSTER"
fi

# kind nodes do not see the host's Docker images: load the image into the cluster
kind load docker-image "$IMAGE" --name "$CLUSTER"

kubectl apply -f deploy/kubernetes/ --context "kind-$CLUSTER"
kubectl rollout status deployment/readmission --context "kind-$CLUSTER" --timeout=180s
kubectl get pods,service,hpa --context "kind-$CLUSTER"

echo
echo "Forward the service, then test it from another terminal:"
echo "  kubectl port-forward service/readmission 9696:80 --context kind-$CLUSTER"
echo "  python test_predict.py"
