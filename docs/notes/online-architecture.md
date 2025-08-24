### The Refined Three-Tier Architecture

This architecture is designed for efficiency, scalability, and maintainability by separating concerns and processing data at the most logical point.

#### Tier 1: The Backend (Core Engine & Analyzer)

This is the heart of the application, responsible for all computation and data processing. It is completely "headless" and has no knowledge of the web or any specific user interface.

1. **The Simulation Engine (`firecast.core`):**

   - **Role:** The raw number-cruncher.
   - **Function:** Its sole responsibility is to take a configuration object and execute a single simulation run, producing a large, detailed dictionary of raw results (e.g., full monthly histories for wealth, returns, etc.).

2. **The Analyzer / Aggregator (`firecast.analysis`):**
   - **Role:** The data processor and summarizer. This is the key improvement.
   - **Function:** This new, shared module takes the list of raw results from _all_ simulation runs. It performs all the heavy data transformations:
     - Calculates summary statistics (success rate, median final wealth, percentiles).
     - Downsamples time-series data (e.g., from monthly to yearly) to create data points suitable for charts.
     - Generates histogram data (bins and counts) for distributions.
   - **Output:** It produces a single, clean, and compact summary object (e.g., a Pydantic model) containing only the aggregated data needed for visualization and reporting.

#### Tier 2: The Web Server (API Layer)

This tier acts as the "controller" or "glue" between the frontend and the backend engine. It handles web-specific tasks and orchestrates the workflow.

- **Role:** The web-facing traffic cop and orchestrator.
- **Function:**
  1. **Exposes API Endpoints:** Provides endpoints like `POST /api/simulate` and `GET /api/results/{task_id}`.
  2. **Manages Jobs:** Receives a configuration from the frontend, validates it using the Pydantic models, and places the simulation job onto a task queue (e.g., Celery) for asynchronous processing.
  3. **Orchestrates Analysis:** Once the simulation workers are done, the web server passes the collected raw results to the **Analyzer module** to get the final, aggregated summary object.
  4. **Serves Data:** It serves two types of data to the frontend:
     - **The Display Payload:** The small, aggregated JSON summary for populating the UI.
     - **The Raw Data Download:** It saves the full, unprocessed results to a file and provides a separate endpoint (`/api/download/{task_id}`) for the user to download if they choose.

#### Tier 3: The Frontend (User Experience)

This is the presentation layer that the user directly interacts with. It is a "dumb" client that is responsible only for display, not for calculation.

- **Role:** The user interface.
- **Function:**
  1. **Configuration:** Provides a user-friendly web form (e.g., in Flutter or HTML/JS) for building a simulation configuration.
  2. **API Communication:** Sends the configuration to the web server's API and polls for the results.
  3. **Visualization:** Receives the **pre-processed, aggregated JSON** from the web server and uses it to render charts (via libraries like `fl_chart`, `Plotly.js` or `Chart.js`) and display summary statistics. It performs no calculations on its own.

This refined architecture ensures that heavy data processing happens efficiently on the backend, code is reused effectively (the Analyzer is used by both the CLI and the web API), and the network payload is minimized, resulting in a faster and more robust application.

## Sizing

### Assumptions

1. **Workload Definition:** "100 concurrent simulations" means the system can handle 100 users submitting their simulation jobs over a short period (e.g., a few minutes). The system will queue these jobs and process them in parallel based on available resources, ensuring a responsive UI and reasonable wait times. It does not mean all 100 simulations (each with 10,000s of runs) are executing on CPU cores at the same instant.
2. **Single Simulation Profile:** A single user's simulation (e.g., 10,000 runs) is CPU-intensive and moderately memory-intensive. Let's estimate it takes **~30-60 seconds** to complete on 4 modern CPU cores and consumes **~2-4 GB of RAM** at its peak (when holding all raw results before aggregation).
3. **User Experience Goal:** A user should receive their results within a few minutes of submission, even during peak load.
4. **Deployment:** The FastAPI web server, Celery (task queue), Redis (message broker), and Celery workers all run in containers on the same VM.

---

### Initial VM Sizing Recommendation

Based on these assumptions, a robust starting point for a single VM would be a **general-purpose, compute-optimized machine**.

| Component   | Recommendation          | Rationale                                                                                                                                                                                                                                                                                                                                                                                                           |
| :---------- | :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CPU**     | **16 vCPUs**            | This is the most critical resource. It allows you to run multiple simulation jobs in parallel. With 16 cores, you could configure your Celery workers to run **4 concurrent user simulations**, with each simulation job being allocated **4 cores** for its internal `ProcessPoolExecutor`. This provides a good balance of throughput and performance for individual jobs.                                        |
| **RAM**     | **64 GB**               | Memory is the second most critical resource. With 4 concurrent jobs, each consuming up to 4 GB of RAM for results, you'd need `4 * 4 = 16 GB` just for the active jobs. Adding memory for the OS, container overhead, Redis, the web server, and a buffer for memory spikes during data aggregation makes 64 GB a safe and reliable choice. It prevents the system from crashing due to Out-Of-Memory (OOM) errors. |
| **Disk**    | **100 GB SSD**          | The application itself is small, but you need space for the OS, container images, logs, and potentially storing raw result files for user download. An SSD is crucial for fast container startup and I/O performance.                                                                                                                                                                                               |
| **Network** | **Standard (1-5 Gbps)** | The application is not network-intensive. The data payloads between the user and the server are small (JSON), and the raw result downloads are infrequent. Standard network bandwidth is sufficient.                                                                                                                                                                                                                |

**Example Cloud Instances:**

- **AWS:** `c5.4xlarge` (16 vCPU, 32 GB RAM) or `m5.4xlarge` (16 vCPU, 64 GB RAM) - the `m5` is a better fit.
- **Google Cloud:** `c3-standard-16` (16 vCPU, 64 GB RAM)
- **Azure:** `Standard_F16s_v2` (16 vCPU, 32 GB RAM) or `Standard_D16s_v5` (16 vCPU, 64 GB RAM)

---

### Scale-Up Strategy

A single, large VM will eventually hit a limit. The professional way to scale is to move from a "vertical" (bigger machine) to a "horizontal" (more machines) architecture.

#### Phase 1: Initial Deployment (Single VM - As described above)

- **Architecture:** All components (Web Server, Redis, Celery Workers) are containerized on one large VM.
- **Pros:** Simple to manage and deploy.
- **Cons:** Single point of failure. Finite capacity.

#### Phase 2: Decoupling the Components (Multiple VMs)

When the single VM can no longer handle the load, the first step is to separate the components onto their own dedicated machines.

1. **Move the Broker:** Relocate Redis to a managed service (like AWS ElastiCache, Google Memorystore) or its own dedicated VM. The message broker is a critical, stateful component and benefits from being managed separately.
2. **Separate Web Server and Workers:**
   - **VM 1 (Web Server):** A smaller VM (e.g., 2-4 vCPUs, 8 GB RAM) dedicated to running the FastAPI web server. Its job is just to handle incoming web requests, so it doesn't need massive resources. You can run multiple instances of this behind a load balancer for high availability.
   - **VM 2+ (Worker Fleet):** One or more large, compute-optimized VMs (like the 16-core machine from Phase 1) dedicated to running the Celery workers. This is where the actual computation happens.

- **Trigger:** You move to this phase when you see the web server becoming unresponsive because the Celery workers are consuming all the CPU/RAM on the shared VM.

#### Phase 3: Horizontal Scaling of Workers (Auto-Scaling)

This is the ultimate goal for a highly scalable system.

1. **Create a Worker Pool:** Instead of manually managing worker VMs, you create an **auto-scaling group** for your Celery workers in the cloud.
2. **Define Scaling Policies:** You configure rules based on the task queue's state. The most common metric is the **number of messages in the queue**.
   - **Scale-Out Rule:** If `number of tasks in queue > 10` for more than 5 minutes, **add a new worker VM** to the pool.
   - **Scale-In Rule:** If `number of tasks in queue < 5` for more than 15 minutes, **remove a worker VM** from the pool.
3. **Container Orchestration:** For maximum efficiency and resilience, you would run your Celery workers inside a container orchestration platform like **Kubernetes (EKS, GKE, AKS)**. This allows you to scale at the container level (pods) instead of the full VM level, which is much faster and more resource-efficient.

By following this phased approach, you can start with a simple, cost-effective deployment and evolve it into a robust, scalable, and resilient architecture as user demand grows.

## Costs

### Core Assumptions for Cost Estimation

1. **Cloud Provider:** AWS (costs are similar on GCP/Azure, but instance names differ).
2. **Region:** `us-east-1` (N. Virginia) - one of the cheapest regions.
3. **Workload Definition:**
   - **100 Concurrent Users:** Represents a "low but steady" load, translating to approximately **100 simulation jobs submitted per hour** during peak times.
   - **1000 Concurrent Users:** Represents a "high load" scenario, translating to **1,000 simulation jobs per hour** during peak times.
4. **Single Job Profile:** One simulation job requires **4 vCPUs** and takes approximately **1 minute** to complete. This is our core unit of compute.
5. **Architecture:** Fully managed Kubernetes (AWS EKS), managed Redis (ElastiCache), and Application Load Balancer.

---

### Cost Breakdown by Component

#### 1. Kubernetes Control Plane (EKS)

This is the "brain" of your Kubernetes cluster, managed by AWS.

- **Cost:** Fixed price per hour, per cluster. ~$0.10/hour.
- **Monthly Cost:** `0.10 * 24 * 30` = **~$73/month**.
- _This cost is the same for both 100 and 1,000 user scenarios, as it's a fixed management fee._

#### 2. Web Server Nodes

These run the FastAPI application. They are not compute-intensive. We need a small, highly-available setup.

- **Instance Type:** `t3.medium` (2 vCPU, 4 GB RAM) - a good, burstable instance type.
- **Setup:** 2 nodes running 24/7 for high availability.
- **Cost:** `2 nodes * ~$0.0416/hour * 24 * 30` = **~$60/month**.
- _This cost is also stable for both scenarios, as the web servers just handle API requests and offload the heavy work._

#### 3. Worker Nodes (Celery) - **The Variable Cost**

This is where auto-scaling happens. The cost is directly proportional to the number of simulation jobs. We'll use compute-optimized instances.

- **Instance Type:** `c5.2xlarge` (8 vCPU, 16 GB RAM). Each can run 2 simulation jobs in parallel (`8 vCPU / 4 vCPU per job`).
- **Cost per hour:** ~$0.34/hour.

**Scenario 1: 100 Concurrent Users (100 jobs/hour)**

- **Compute Needed:** `100 jobs/hour * 1 minute/job` = 100 minutes of compute time per hour.
- **Nodes Needed:** Each `c5.2xlarge` provides `2 jobs/node * 60 minutes` = 120 job-minutes of capacity per hour.
- **Utilization:** We need 100 out of 120 available job-minutes. This means we need **one `c5.2xlarge` node running ~83% of the time.**
- **Monthly Cost:** `1 node * $0.34/hour * 0.83 * 24 * 30` = **~$203/month**.

**Scenario 2: 1,000 Concurrent Users (1,000 jobs/hour)**

- **Compute Needed:** `1,000 jobs/hour * 1 minute/job` = 1,000 minutes of compute time per hour.
- **Nodes Needed:** `1,000 job-minutes / 120 job-minutes per node` = **~8.33 nodes**. The auto-scaler will dynamically scale up to 8 or 9 nodes during peak hours and scale down to 0 or 1 during off-hours. We'll estimate an average of 8.33 nodes running constantly for simplicity.
- **Monthly Cost:** `8.33 nodes * $0.34/hour * 24 * 30` = **~$2,039/month**.

#### 4. Managed Redis (ElastiCache)

This is the message broker for Celery. It doesn't need to be large, but it should be reliable.

- **Instance Type:** `cache.t3.small` (replicated across two availability zones for high availability).
- **Cost:** `2 nodes * ~$0.034/hour * 24 * 30` = **~$49/month**.
- _This cost is stable for both scenarios. The queue depth won't stress a small Redis instance._

#### 5. Application Load Balancer (ALB)

This distributes traffic to your web server nodes.

- **Cost:** Fixed hourly cost + data processing fee.
- **Monthly Cost:** `~$0.0225/hour * 24 * 30` + data fees = **~$20/month**.
- _This cost is also stable._

#### 6. Data Transfer & Storage

This includes traffic from users, container images in ECR, and logs.

- **Estimate:** Typically a small percentage of the total cost for an application like this.
- **Monthly Cost:** **~$10 - $50/month**.

---

### Total Estimated Monthly Cost Summary

| Component                        | 100 Concurrent Users | 1,000 Concurrent Users | Rationale                                                   |
| :------------------------------- | :------------------- | :--------------------- | :---------------------------------------------------------- |
| EKS Control Plane                | ~$73                 | ~$73                   | Fixed management fee.                                       |
| Web Server Nodes                 | ~$60                 | ~$60                   | Stable base for API requests.                               |
| **Worker Nodes (Auto-Scaled)**   | **~$203**            | **~$2,039**            | **This is the core variable cost, directly tied to usage.** |
| Managed Redis (ElastiCache)      | ~$49                 | ~$49                   | Stable infrastructure component.                            |
| Load Balancer                    | ~$20                 | ~$20                   | Stable infrastructure component.                            |
| Data Transfer & Other            | ~$25                 | ~$75                   | Minor cost, scales slightly with usage.                     |
| **Total Estimated Monthly Cost** | **~$430 / month**    | **~$2,316 / month**    |                                                             |

### Key Takeaways

- **Variable Costs Dominate:** The cost is overwhelmingly driven by the actual compute usage of the worker nodes. This is the power of auto-scaling: you only pay for what you use.
- **Base Cost:** There is a fixed "cost of admission" of around **$200-$250/month** just to have the highly-available infrastructure (EKS, ALB, Redis, Web Servers) running, even with zero traffic.
- **Optimization:** If costs become a concern, you can leverage **AWS Spot Instances** for the worker nodes. Spot Instances use spare AWS capacity at a discount of up to 90%. Since your simulation jobs are stateless and can be retried if a Spot Instance is reclaimed, they are a perfect workload for this, potentially cutting your worker node costs by 70% or more.
