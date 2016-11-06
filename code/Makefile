SHELL = /bin/bash

all:
	@echo "Hello NNs!"

images/perceptron-END.png:
	python perceptron.py

animation.gif: images/perceptron-END.png
	# convert to gif
	for i in $$(find images -name "*png"); do [ ! -e "$${i%.png}.gif" ] && convert $$i $${i%.png}.gif; done
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
		images/perceptron-END.gif > animation.gif

clean:
	rm -rf images/
	rm -f animation.gif
