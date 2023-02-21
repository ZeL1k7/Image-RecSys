data_preprocessing:
	mkdir -p data/raw data/interim data/processed
	curl https://drive.google.com/file/d/1bYC_a9TyXfybd-dCBBoeKaz0Ny4UO0fX/view?usp=share_link -o data/raw/dataset.zip
	unzip data/raw/dataset.zip	

make_embeddings:
	python data/raw/ data/interim/

get_recommendations:
	python data/raw/ data/interim/ data/processed/

postprocessing:
	python data/raw/ data/processed/ data/processed/

