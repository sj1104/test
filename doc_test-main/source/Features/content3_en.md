Memory-bounded Training
====================================

Since the trend of Deep Neural Networks (DNNs) is toward deeper and larger, performing DNNs training on existing accelerators (e.g., GPUs) is challenging due to their limited device memory capacity. Existing memory management systems reduce the memory footprint via tensor offloading and recomputing. However, the coarse-grained, one-tensor-at-a-time memory management often incurs high peak memory usage and cannot fully utilize available hardware resources. 

We propose a fine-grained DNN memory management system that breaks apart memory bottlenecks of training and greatly improves the efficiency of memory-optimized DNNs training. Evaluations on differen DNN models show that compared to vDNN and SuperNeurons, our system can achieve maximum batch size up to 10.5× and 3.1×, and throughput up to 4.7× and 2.7× under the same memory over-subscription, respectively.



<div class="warning">
<em>Our paper is under-reviewing and we will release the full details as soon as possible.</em>
</div>