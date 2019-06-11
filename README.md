# object-localization
run `docker build --tag=object_localizer .` <br>
run `docker run -p 4000:80 object_localizer` <br>
<h4> OR </h4>
run `wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz` <br>
run `wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz` <br>
run `tar xf images.tar.gz` <br>
run `tar xf annotations.tar.gz` <br>
run `mv annotations/xmls/* images/` <br>
run `python3 generate_dataset.py` <br>
run `python3 train.py` <br>

