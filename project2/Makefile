roemelle_gru:
	./experiments/roemelle_gru.sh

roemelle_gru_add_attn:
	./experiments/roemelle_gru_add_attn.sh

roemelle_gru_mult_attn:
	./experiments/roemelle_gru_mult_attn.sh

roemelle_bigru:
	./experiments/roemelle_bigru.sh

roemelle_bigru_add_attn:
	./experiments/roemelle_bigru_add_attn.sh

roemelle_bigru_mult_attn:
	./experiments/roemelle_bigru_mult_attn.sh

roemelle_lstm:
	./experiments/roemelle_lstm.sh

roemelle_lstm_add_attn:
	./experiments/roemelle_lstm_add_attn.sh

roemelle_lstm_mult_attn:
	./experiments/roemelle_lstm_mult_attn.sh

roemelle_bilstm:
	./experiments/roemelle_bilstm.sh

roemelle_bilstm_add_attn:
	./experiments/roemelle_bilstm_add_attn.sh

roemelle_bilstm_mult_attn:
	./experiments/roemelle_bilstm_mult_attn.sh

roemelle_vanilla:
	./experiments/roemelle_vanilla.sh

srini_ffn_fc_gru:
	./experiments/srini_ffn_fc_gru.sh

srini_ffn_fc_lstm:
	./experiments/srini_ffn_fc_lstm.sh

srini_ffn_ls:
	./experiments/srini_ffn_ls.sh

requirements:
	python -m pip install -r requirements.txt --user
	python -c 'import nltk; nltk.download("punkt")'
	cd data && ./download_data.sh
