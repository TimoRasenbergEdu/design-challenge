IMAGE_NAME=rl
LOCAL_DIR=~/Documents/Fontys/S7\ \(AI\)/Design\ Challenge
SCRIPT_NAME=rl/ddqn-qbert.py

.PHONY: build
build:
	sudo docker build -t $(IMAGE_NAME) .

.PHONY: run
run:
	sudo docker run --gpus all -it --rm -v $(LOCAL_DIR):/app $(IMAGE_NAME) python /app/$(SCRIPT_NAME)

.PHONY: clean
clean:
	sudo docker image prune -f
