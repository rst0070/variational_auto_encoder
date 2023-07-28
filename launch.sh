sudo docker container rm vae
sudo docker build -t vae .
sudo nvidia-docker run \
    --name vae \
    --shm-size=50gb \
    -v /home/rst/dataset/celeba/img_align_celeba:/data/celeba \
    -v /home/rst/workspace/vae/result:/result \
    -t vae:latest