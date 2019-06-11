roemmele_gru:
	./experiments/roemmele_gru.sh

roemmele_gru_add_attn:
	./experiments/roemmele_gru_add_attn.sh

roemmele_gru_mult_attn:
	./experiments/roemmele_gru_mult_attn.sh

roemmele_bigru:
	./experiments/roemmele_bigru.sh

roemmele_bigru_add_attn:
	./experiments/roemmele_bigru_add_attn.sh

roemmele_bigru_mult_attn:
	./experiments/roemmele_bigru_mult_attn.sh

roemmele_lstm:
	./experiments/roemmele_lstm.sh

roemmele_lstm_add_attn:
	./experiments/roemmele_lstm_add_attn.sh

roemmele_lstm_mult_attn:
	./experiments/roemmele_lstm_mult_attn.sh

roemmele_bilstm:
	./experiments/roemmele_bilstm.sh

roemmele_bilstm_add_attn:
	./experiments/roemmele_bilstm_add_attn.sh

roemmele_bilstm_mult_attn:
	./experiments/roemmele_bilstm_mult_attn.sh

roemmele_vanilla:
	./experiments/roemmele_vanilla.sh

srini_ffn_fc_gru:
	./experiments/srini_ffn_fc_gru.sh

srini_ffn_fc_lstm:
	./experiments/srini_ffn_fc_lstm.sh

srini_ffn_ls:
	./experiments/srini_ffn_ls.sh

universal_all:
	./experiments/generate_embeddings_use.sh
	./experiments/train_use.sh

universal_all_val:
	./experiments/generate_embeddings_use.sh
	./experiments/val_use.sh

skipthoughts_all_val:
	./experiments/generate_embeddings_st.sh
	./experiments/val_st.sh

requirements:
	python -m pip install -r requirements.txt --user
	python -c 'import nltk; nltk.download("punkt")'
	cd data && ./download_data.sh
