.PHONY: build train-wall train-self train-fruit train-multi play play-wall play-self play-fruit play-multi clean

TRAIN_BIN := bin/train
PLAY_BIN := bin/play

build:
	mkdir -p bin artifacts runs
	go build -o $(TRAIN_BIN) ./cmd/train
	go build -o $(PLAY_BIN) ./cmd/play

train-wall: build
	$(TRAIN_BIN) -config configs/wall.yaml

train-self: build
	$(TRAIN_BIN) -config configs/self.yaml

train-fruit: build
	$(TRAIN_BIN) -config configs/fruit.yaml

train-multi: build
	$(TRAIN_BIN) -config configs/multi.yaml

# Play with the last trained champion (specify config to match the trained model)
# Use -no-timeout to let it play until it dies
play: build
	$(PLAY_BIN) -config configs/wall.yaml -champion artifacts/champion_final.json -no-timeout

play-wall: build
	$(PLAY_BIN) -config configs/wall.yaml -champion artifacts/champion_final.json -no-timeout

play-self: build
	$(PLAY_BIN) -config configs/self.yaml -champion artifacts/champion_final.json -no-timeout

play-fruit: build
	$(PLAY_BIN) -config configs/fruit.yaml -champion artifacts/champion_final.json -no-timeout

play-multi: build
	$(PLAY_BIN) -config configs/multi.yaml -champion artifacts/champion_final.json -no-timeout

clean:
	rm -rf bin artifacts runs

