alias submit="bsub -n 4 -W 24:00 -R \"rusage[mem=18000, ngpus_excl_p=1]\" python"
mkdir -p logs/

ENCODER="--encoder_type universal"

for RNN_TYPE in "gru" "lstm"
do
	submit val_rnn.py $ENCODER --rnn_type $RNN_TYPE &
	submit val_ffn.py $ENCODER --rnn_type $RNN_TYPE &
	submit val_birnn.py $ENCODER --rnn_type $RNN_TYPE &

	for ATT_TYPE in "multiplicative" "additive"
	do
		submit val_rnn.py $ENCODER --rnn_type $RNN_TYPE --use_attn --attn_type $ATT_TYPE &
		submit val_birnn.py $ENCODER --rnn_type $RNN_TYPE --use_attn --attn_type $ATT_TYPE  &
	done
done