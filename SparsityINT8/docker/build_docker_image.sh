# Copy relevant files to docker's build context
cp ../*.py ./scripts

# Build docker image
IMAGE_NAME="Name this image as you wish!!"
docker build --tag $IMAGE_NAME --network=host -f Dockerfile .

## Change tag name if needed
# docker tag $IMAGE_NAME:latest $IMAGE_NAME:23.05
