# object-localization
run `wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz`
run `wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz`
run `tar xf images.tar.gz`
run `tar xf annotations.tar.gz`
run `mv annotations/xmls/* images/`
run `python3 generate_dataset.py`
run `python train.py`

