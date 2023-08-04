sudo docker container rm vae2
sudo docker build -t vae .
sudo nvidia-docker run \
    --name vae2 \
    --shm-size=50gb \
    -v /home/rst/dataset/celeba/img_align_celeba:/data/celeba \
    -v /home/rst/workspace/vae/result:/result \
    -t vae:latest