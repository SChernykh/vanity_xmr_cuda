# The CUDA version matters to some degree, and must at least work with the
# driver used in the container host.
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:12.8.1-runtime-ubuntu24.04

FROM $BASE_CUDA_DEV_CONTAINER AS build

# Bring in source
WORKDIR /app
COPY . /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
   apt install -y cmake && \
   rm -rf /var/lib/apt/lists/*

RUN mkdir build && cd build && cmake .. && make -j$(nproc)

# The runtime image is approximately half the size of the `build` image.
FROM $BASE_CUDA_RUN_CONTAINER

COPY --from=build /app/build/vanity_xmr_cuda /usr/bin 

ENTRYPOINT ["/usr/bin/vanity_xmr_cuda"]
