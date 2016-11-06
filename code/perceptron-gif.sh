#/bin/bash

for i in perceptron-00*.png; do
    [ ! -e "${i%.png}.gif" ] && convert $i ${i%.png}.gif
done

convert perceptron-final.png perceptron-final.gif
F=perceptron-final.gif
gifsicle --colors 16 --delay=35 --loop perceptron-00*.gif $F $F $F $F $F $F $F $F $F $F > animation.gif
