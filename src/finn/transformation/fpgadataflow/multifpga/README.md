# Multi-FPGA in FINN+
Multi-FPGA usage in FINN+ is implemented purely in the second half of the FINN-flow. This means you can use FINN as normal, and even use the same steps - Multi-FPGA is implicitly switched on by providing a partitioning configuration in your flow config file. As soon as these settings are detected, FINN+ in the background switches the second half of the flow to the Multi-FPGA specific steps and transformations.

## Multi-FPGA specific steps
There are 3-4 Multi-FPGA specific steps that need to be executed before synthesis.

### 1. Partitioning
To decide which layers/nodes are put onto which device, partitioning is done. If an existing assignment exists, this can be passed as an argument. If no assignment exists yet, the dataflow graph is converted into a (M)-ILP model. This model is then solved to assign every node to one device ID.

### 2. Multi-FPGA StreamingDataflowPartitions
As soon as the assignment of the device IDs is complete, consecutive nodes with the same device ID are grouped together into _StreamingDataflowPartition_ nodes (_SDPs_). These are meta-nodes that act as containers for subgraphs. The SDP is then assigned the same device ID as all of the nodes it contains.

### 3. Metadata Creation
At this point, an internal model of the finished accelerator design is created. It stores which SDPs are connected with which other SDPs on which devices, where the communication kernels are placed, how many ports each design uses, if a port is used for TX, RX or both, etc. This metadata is saved alongside the ONNX model.

### 4. Communication Kernel Preparation
At this point, depending on which communication methodology is used, custom preparation steps can be done. For example, the _AuroraFlow_ kernel needs to be configured and packaged into an XO file to be used at the linking stage. Such preparations can be done in this step (but are not required).

### (Synthesis)
After everything is done, the SDPs are packaged into XOs as well (Vitis flow only) and the linker configuration is created. For Multi-FPGA, an additional transformation will be executed. This additional transformation modifies the linker configuration to, for example, instantiate the communication kernel and connect its stream interface with the compute kernel.
