export BIOHOME=/home/bb/valejose/valejose
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
rna=${1}_RNA.fa
#prot=${1}_PROTEIN.fa
prot=${1}_micropeptides.fa
python design/design_bioseq2seq.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --attribution_mode $2 --num_gpus 0  --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $1 --tgt_class $3 --tgt_pos $4 --rank 0 --sample_size 32 --minibatch_size 16


