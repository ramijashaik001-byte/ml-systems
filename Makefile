install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m unittest discover -s tests

run:
	python examples/serve.py

docker-build:
	docker build -t nexusml:latest .

docker-run:
	docker run -p 8080:8080 nexusml:latest
