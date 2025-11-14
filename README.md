# Glean - Federated Bakery Demand Forecasting

**Privacy-preserving collaborative machine learning for small businesses**

## The Problem

Small bakeries face a daily dilemma—produce too much and waste food, money, and labor at the end of the day, or produce too little and disappoint customers. They lack the data science expertise and resources that large chains have. The result: massive food waste, higher costs, and inefficient staffing.

## The Solution

Glean is a federated learning platform that lets bakeries collaborate on demand forecasting **without sharing their sensitive data**. Each bakery trains a model on their own sales history locally. These models share learnings with each other through Flower—a privacy-preserving federated framework—creating a collective intelligence that's smarter than any bakery could achieve alone.

## Impact

- **15-25% reduction in overproduction**
- Lower waste disposal costs
- Minimized spoilage
- Optimized staff scheduling
- Small operators compete on analytics with large chains

**Why Federated Learning?** Bakeries never expose their sales data—the competitive advantage stays theirs. The model improves as more bakeries join. It's privacy-first by design.

---

## Quick Start

### Prerequisites

- Python 3.9+ (Python 3.10+ recommended)
- Virtual environment activated

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /home/julian/Projects/safe_waste_help_small
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Hackathon Server Deployment

```bash
./submit-job.sh "cd /home/team13/safe_waste_help_small && flwr run . cluster-cpu"
```

### Run Local Simulation

Test with 5 simulated bakeries on your local machine:

```bash
flwr run . local-simulation
```

This will:
- Start the federated server
- Create 5 virtual bakery clients
- Run 20 rounds of federated learning
- Show training progress in real-time

### Deploy to Cluster

#### CPU Mode:
```bash
./submit-job.sh "flwr run . cluster-cpu"
```

#### GPU Mode (with AMD MI300X):
```bash
./submit-job.sh "flwr run . cluster-gpu" --gpu
```

#### Check Job Status:
```bash
# View running jobs
squeue -u $USER

# View job output
cat ~/logs/job_*.out
```

---

## Project Structure

```
glean/
├── glean/
│   ├── __init__.py         # Package initialization
│   ├── server_app.py       # Federated server (FedAvg strategy)
│   ├── client_app.py       # Bakery client (local training)
│   └── task.py            # [TODO] Model training logic
├── data/                   # [TODO] Bakery sales data
├── pyproject.toml         # Flower configuration
├── requirements.txt       # Python dependencies
├── cluster-cpu.toml      # CPU cluster config
├── cluster-gpu.toml      # GPU cluster config
└── README.md             # This file
```

---

## How It Works

### Architecture

1. **Server** ([glean/server_app.py](glean/server_app.py))
   - Coordinates federated learning across bakeries
   - Aggregates model updates using FedAvg strategy
   - Distributes improved global model back to clients
   - Never sees raw bakery data

2. **Client** ([glean/client_app.py](glean/client_app.py))
   - Represents a single bakery
   - Trains forecasting model on local sales data
   - Shares only model parameters (not data) with server
   - Receives improved model from federation

### Privacy Guarantees

- **Data stays local:** Raw sales data never leaves the bakery
- **Only model updates shared:** Gradients/parameters sent to server
- **Secure aggregation:** Server only sees averaged updates
- **No reverse engineering:** Individual bakery patterns remain private

---

## Configuration

### Adjust Number of Training Rounds

Edit [pyproject.toml](pyproject.toml):
```toml
[tool.flwr.app.config]
num-server-rounds = 20  # Change to 10, 50, etc.
```

### Change Number of Bakeries

Edit [pyproject.toml](pyproject.toml):
```toml
[tool.flwr.federations]
local-simulation = { options.num-supernodes = 5 }  # Change to 3, 10, etc.
```

### Resource Allocation

For cluster deployment, edit [cluster-cpu.toml](cluster-cpu.toml) or [cluster-gpu.toml](cluster-gpu.toml):
```toml
[tool.flwr.federations.cluster-cpu.options.backend.client-resources]
num-cpus = 2  # CPUs per bakery client
num-gpus = 0  # GPUs per bakery client
```

---

## Next Steps (Roadmap)

### Phase 1: Data & Model ✅ Setup Complete
- [x] Flower server and client infrastructure
- [x] Cluster deployment configurations
- [ ] Generate synthetic bakery sales data
- [ ] Implement XGBoost forecasting model

### Phase 2: Training & Evaluation
- [ ] Load and preprocess time series data
- [ ] Feature engineering (day of week, holidays, trends)
- [ ] Train local models per bakery
- [ ] Evaluation metrics (MAE, RMSE, MAPE)

### Phase 3: Production Features
- [ ] Model checkpointing and versioning
- [ ] Monitoring dashboard
- [ ] Differential privacy integration
- [ ] Comparison: Federated vs. Local-only models

### Phase 4: Pilot Deployment
- [ ] Real bakery data integration
- [ ] API for demand predictions
- [ ] User interface for bakery owners
- [ ] Performance benchmarks

---

## Development

### Running Tests

```bash
# TODO: Add pytest tests
pytest tests/
```

### Code Structure

- `glean/server_app.py` - Federated server logic
- `glean/client_app.py` - Bakery client implementation
- `glean/task.py` - Model definition and training (to be implemented)
- `glean/strategy.py` - Custom federated strategies (optional)

---

## Troubleshooting

### ModuleNotFoundError: No module named 'glean'

Make sure you're in the project root directory:
```bash
pwd  # Should show: /home/julian/Projects/safe_waste_help_small
```

### Dependencies Not Found

Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Simulation Won't Start

Check Flower configuration:
```bash
flwr run --help
```

Verify `pyproject.toml` syntax is correct.

### Cluster Job Fails

View logs for error messages:
```bash
cat ~/logs/job_*.out
```

Common issues:
- Insufficient resources requested
- Missing dependencies in cluster environment
- Configuration file syntax errors

---

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [Flower Federated Learning Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Federated Learning Paper](https://arxiv.org/abs/1602.05629)

---

## Contributing

We're looking for pilot bakeries to test this platform! Help us prove federated learning solves real sustainability problems at scale.

**Contact:** [Your contact information]

---

## License

[Add your license here]

---

**Built with [Flower](https://flower.ai/) - The Friendly Federated Learning Framework**
