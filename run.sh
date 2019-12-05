datasets=("cora")
model_categorys=("ours" "base")
use_intermediate_embeddings=(0 1)
use_linear_combs=(0 1)
n_hiddens=(64 128 256)
for dataset in ${datasets[@]}; do
	for model_category in ${model_categorys[@]}; do
		for use_intermediate_embedding in ${use_intermediate_embeddings[@]}; do
			for use_linear_comb in ${use_linear_combs[@]}; do
				for n_hidden in ${n_hiddens[@]}; do
						python3 main.py --data $dataset --use_intermediate_embedding $use_intermediate_embedding --use_linear_comb $use_linear_comb --n_hidden $n_hidden --n_epochs 1 --model_category $model_category
				done
			done
		done
	done
done