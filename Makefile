# Install python dependencies
requirements:
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt

# Create requirements file
create-requirements:
	pip freeze > requirements.txt

# Delete python compiled files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Make commands
data:
	python src/data/make_dataset.py

features:
	python src/features/build_features.py

train:
	python src/models/train_model.py

predict:
	python src/models/predict_model.py

visualization:
	python src/visualization/visualize.py

all: requirements data features train predict visualization

