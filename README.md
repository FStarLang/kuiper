# Kuiper: GPU Kernel Verification in Pulse

## Overview
Kuiper is a domain-specific language (DSL) designed for programming and verifying safe GPU kernels. Built on F* and Pulse, Kuiper offers a high-level abstraction of CUDA, allowing developers to write optimized, low-level GPU kernels while ensuring safety and functional correctness. Kuiper enables verification of essential properties like data race freedom, termination, and functional correctness, which are crucial for reliable and efficient GPU programming.

## Key Features
- **GPU Kernel Programming**: Kuiper provides an intuitive DSL for programming GPU kernels, leveraging Pulse’s separation logic to ensure safe parallel and asynchronous code execution.
- **Verification**: The language supports verification of properties like data race freedom, termination, and functional correctness, helping developers write error-free code.
- **Single GPU Focus**: Initially, Kuiper focuses on single GPU kernels, involving reasoning about parallelism and synchronization within a single device.
- **Extensibility**: The long-term goal is to extend Kuiper’s capabilities to multi-node communication (e.g., MSCCL) and heterogeneous CPU-GPU programs (e.g., SYCL).
- **LLM Integration**: Future versions of Kuiper will integrate Large Language Models (LLMs) to assist in writing and verifying GPU kernels, simplifying the process for developers.

## Use Cases
- **Safe GPU Programming**: Kuiper allows for the development of optimized GPU kernels while ensuring that the code is free of common errors, such as data races and incorrect behavior.
- **Verification of GPU Code**: Kuiper can be used to formally verify the correctness of GPU kernels before deploying them, reducing the time spent debugging complex issues.
- **Complex GPU-CPU Systems**: Kuiper lays the foundation for verifying communication and computation between CPU and GPU, making it a valuable tool for developers working on heterogeneous systems.

## Getting Started

### Prerequisites
1. **Docker**: You will need Docker installed on your system to run Kuiper in a containerized environment.
2. **Git**: Ensure that Git is installed to clone the repository.

### Clone the Kuiper Repository
First, clone the Kuiper repository with submodules included:
```bash
git clone --recurse-submodules https://github.com/mtzguido/kuiper.git
cd kuiper
git submodule init
git submodule update --depth 1
```

### Pull the Docker Image
Pull the `pulse-cuda-devcontainer` image from Docker Hub:
```bash
docker pull mtzguido/pulse-cuda-devcontainer
```

### Run the Docker Container
Start the Docker container with the following command, ensuring it has the necessary privileges and access to GPUs:
```bash
docker run -itd --name=pulse-cuda--privileged --ipc=host --net=host --gpus=all -w /root --ulimit memlock=-1:-1 -v $HOME:$HOME --user root mtzguido/pulse-cuda-devcontainer bash
```
This command will:
- Run the container interactively (`-it`)
- Detach it (`-d`) so that it runs in the background
- Use the host's network (`--net=host`) and IPC namespace (`--ipc=host`)
- Allocate all GPUs to the container (`--gpus=all`)
- Set the working directory to `/root`
- Mount your home directory inside the container (`-v $HOME:$HOME`)

### Access the Docker Container
Once the container is running, access it using:
```bash
docker exec -it pulse-cuda bash
```
This will open a bash shell inside the running Docker container.

### Compile Kuiper
Once inside the Docker container, navigate to the Kuiper directory:
```bash
cd /home/<your_user>/kuiper
```
Replace `<your_user>` with your actual username inside the container.

Now, compile Kuiper using `make`:
```bash
make
```
This will build the Kuiper project and prepare it for use.

### Run the Benchmarks
To run the benchmarks and test the compiled kernels, execute the provided script:
```bash
./scripts/bench.sh
```
This will run the benchmarks and provide performance and correctness outputs for the compiled code.

## Contributing
Kuiper is still in its early stages, and contributions are welcome! If you have suggestions, improvements, or want to help extend Kuiper’s capabilities, feel free to open issues or submit pull requests.

## Future Work
- **Multi-GPU Support**: Extending Kuiper to handle multi-GPU systems, including communication between nodes.
- **LLM-Enhanced Programming**: Integrating LLMs into the development process to assist in writing and verifying GPU kernels.
- **Heterogeneous Computing**: Expanding Kuiper to support verification of programs across CPU and GPU, including support for frameworks like SYCL.

## License
Kuiper is licensed under the <UPDATE_WITH_LICENSE> License. See the LICENSE file for more details.


