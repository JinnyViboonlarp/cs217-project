# CS217 Project - An Annotation Platform for Cooking-Domain NER

This repository is a collaborated effort of Jinny Viboonlarp (jviboonlarp@brandeis.edu), Theo Solter (theosolter@brandeis.edu), and Hasan Allahyarov (hasanallahyarov@brandeis.edu). Please feel free to contact us if there is any problem running the codes.

This repo is mostly about running the web app using an already-trained-by-us NER model. For the details about how we trained that model and experimented with different model architectures, please visit [our "NLP component" repo](https://github.com/JinnyViboonlarp/cs217-training-cooking-ner/tree/main)

### Running the web application

First, please make sure that every package in `requirements.txt` is installed in your environment.

To run the code, please clone this repo and run `python app.py`. Then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Running the web application in Docker mode

Please run
```
docker build -t project .
docker run --rm -it -p 5000:5000 -v <your_path_to_the_folder_containing_this_repo>:/app/ project
```
For example, if this repo is cloned to "D:\cs217-project", your command would be
```
docker run --rm -it -p 5000:5000 -v D:\cs217-project:/app/ project
```
Then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

Note that building a docker image (i.e. `docker build -t project .`) might take many minutes to run on the first time, since some of the packages, like `torch`, is quite large.

### Removing the database

To remove/reset the database, please delete the sqlite file in the folder `instance` and restart the web application.


