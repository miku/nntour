SHELL = /bin/bash

all:
	@echo "Hello NNs!"

# Perceptron
images/perceptron-END.png:
	python perceptron.py

perceptron.gif: images/perceptron-END.png
	# convert to gif
	for i in $$(find images -name "perceptron-*png"); do [ ! -e "$${i%.png}.gif" ] && convert $$i $${i%.png}.gif; done
	# convert final image to gif
	convert images/perceptron-END.png images/perceptron-END.gif
	# concat, repeat last frame ten times
	gifsicle --colors 16 --delay=35 --loop images/perceptron-00*.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif \
		images/perceptron-END.gif > perceptron.gif

# Random weights
images/random-END.png:
	python randomweights.py

random.gif: images/random-END.png
	# convert to gif
	for i in $$(find images -name "random-*png"); do [ ! -e "$${i%.png}.gif" ] && convert $$i $${i%.png}.gif; done
	# convert final image to gif
	convert images/random-END.png images/random-END.gif
	# concat, repeat last frame ten times
	gifsicle --colors 16 --delay=35 --loop images/random-00*.gif \
		images/random-END.gif \
		images/random-END.gif \
		images/random-END.gif \
		images/random-END.gif \
		images/random-END.gif \
		images/random-END.gif \
		images/random-END.gif \
		images/random-END.gif > random.gif


# Pocket algorithm
images/pocket-END.png:
	python pocket.py

pocket.gif: images/pocket-END.png
	# convert to gif
	for i in $$(find images -name "pocket-*png"); do [ ! -e "$${i%.png}.gif" ] && convert $$i $${i%.png}.gif; done
	# convert final image to gif
	convert images/pocket-END.png images/pocket-END.gif
	# concat, repeat last frame ten times
	gifsicle --colors 16 --delay=35 --loop images/pocket-00*.gif \
		images/pocket-END.gif \
		images/pocket-END.gif \
		images/pocket-END.gif \
		images/pocket-END.gif \
		images/pocket-END.gif \
		images/pocket-END.gif \
		images/pocket-END.gif \
		images/pocket-END.gif > pocket.gif



# XOR-ish algorithm
images/xorish-END.png:
	python xorish.py

xorish.gif: images/xorish-END.png
	# convert to gif
	for i in $$(find images -name "xorish-*png"); do [ ! -e "$${i%.png}.gif" ] && convert $$i $${i%.png}.gif; done
	# convert final image to gif
	convert images/xorish-END.png images/xorish-END.gif
	# concat, repeat last frame ten times
	gifsicle --colors 16 --delay=35 --loop images/xorish-00*.gif \
		images/xorish-END.gif \
		images/xorish-END.gif \
		images/xorish-END.gif \
		images/xorish-END.gif \
		images/xorish-END.gif \
		images/xorish-END.gif \
		images/xorish-END.gif \
		images/xorish-END.gif > xorish.gif

clean:
	rm -rf images
	rm -f perceptron.gif
	rm -f random.gif
	rm -f pocket.gif
	rm -f xorish.gif