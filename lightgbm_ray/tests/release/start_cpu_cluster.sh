if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

export LIGHTGBM_RAY_PACKAGE="${LIGHTGBM_RAY_PACKAGE:-lightgbm_ray}"
export NUM_WORKERS="${NUM_WORKERS:-3}"

SESSION_NAME=${SESSION_NAME:-lightgbm_ray_release_cpu_$(date +%s)}

echo "Starting GPU cluster with ${NUM_WORKERS} worker nodes (plus the head node)"
echo "This will install lightgbm_ray using the following package: ${LIGHTGBM_RAY_PACKAGE}"

CMD="anyscale up --cloud-name anyscale_default_cloud --config cluster_cpu.yaml ${SESSION_NAME}"

echo "Running: ${CMD}"
${CMD}
