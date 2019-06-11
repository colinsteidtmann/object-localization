# object-localization
<h4> First get data </h4> <hr>
run `wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz` <br>
run `wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz` <br>
run `tar xf images.tar.gz` <br>
run `tar xf annotations.tar.gz` <br>
run `mv annotations/xmls/* images/` <br>
run `python3 generate_dataset.py` <br> <br>

<h4> Next, run Docker (or python3 train.py if you got tensorflow and things set up) </h4> <hr>
run `docker build --tag=object_localizer .` <br>
run `docker run -p 4000:80 object_localizer` for cpu (not recommended)<br>
run `docker run --runtime=nvidia -p 4000:80 object_localizer` for nvidia gpu (recommended)<br>

