SHELL = /bin/bash

all:
	@echo "Hello NNs!"

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

clean:
	rm -rf images
	rm -f perceptron.gif
	rm -f random.gif
