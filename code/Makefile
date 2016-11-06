SHELL = /bin/bash

all:
	@echo "Hello NNs!"

images:
	python perceptron.py

animation.gif:
	for i in $$(find images -name "*png"); do [ ! -e "$${i%.png}.gif" ] && convert $$i $${i%.png}.gif; done
	convert images/perceptron-END.png images/perceptron-END.gif
	F=perceptron-final.gif gifsicle --colors 16 --delay=35 --loop perceptron-00*.gif \
		$$F $$F $$F $$F $$F $$F $$F $$F $$F $$F > animation.gif

clean:
	rm -rf images/
	rm -f animation.gif