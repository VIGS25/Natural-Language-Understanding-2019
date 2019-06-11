alias submit="bsub -n 4 -W 24:00 -R \"rusage[mem=18000, ngpus_excl_p=1]\" python"
ln -s ../story_cloze story_cloze
mkdir -p logs/

ENCODER="--embed_mode both"

for RNN_TYPE in "gru" "lstm"
do
	submit story_cloze.train.train_rnn $ENCODER --rnn_type $RNN_TYPE &
	submit story_cloze.train.train_ffn $ENCODER --rnn_type $RNN_TYPE & --embed_mode both
  submit story_cloze.train.train_ffn $ENCODER --rnn_type $RNN_TYPE & --embed_mode both --input_mode last_sentence
	submit story_cloze.train.train_birnn $ENCODER --rnn_type $RNN_TYPE &

	for ATT_TYPE in "multiplicative" "additive"
	do
		submit story_cloze.train.train_rnn $ENCODER --rnn_type $RNN_TYPE --use_attn --attn_type $ATT_TYPE &
		submit story_cloze.train.train_birnn $ENCODER --rnn_type $RNN_TYPE --use_attn --attn_type $ATT_TYPE  &
	done
done
