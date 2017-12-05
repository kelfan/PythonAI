# build Python working environment by docker
## sources
https://hub.docker.com/explore/

## ermaker/keras-jupyter = Jupyter with Keras (with Theano backend and TensorFlow backend)
https://hub.docker.com/r/ermaker/keras-jupyter/
```shell
docker run -d -p 8888:8888 ermaker/keras-jupyter
# With Tensorflow: (See this for more information)

docker run -d -p 8888:8888 -e KERAS_BACKEND=tensorflow ermaker/keras-jupyter
# For persistent storage:

docker run -d -p 8888:8888 -v /notebook:/notebook ermaker/keras-jupyter
# Just browse localhost:8888 and write code with Keras!
```
