for seed in {1..20}; do
	for gnn in GATv2Conv RGCN_8x32_ROOT_SHARED; do
		python analyze.py ${gnn} scatter ${seed}
		python analyze.py ${gnn} clustermap ${seed}
	done
done