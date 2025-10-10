import numpy as np

class StateNormalizer:
    """Handles state normalization with running statistics"""
    
    def __init__(self, state_dim, epsilon=1e-8, n_cells=10):
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.n_cells = n_cells

        # Simulation features normalization bounds (first 18 features)
        self.simulation_bounds = {
            'totalCells': [1, 50],               # number of cells
            'totalUEs': [1, 200],                # number of UEs
            'simTime': [600, 3600],              # simulation time
            'timeStep': [1, 10],                 # time step
            'timeProgress': [0, 1],              # progress ratio
            'carrierFrequency': [700e6, 6e9],    # frequency Hz
            'isd': [100, 2000],                  # inter-site distance
            'minTxPower': [0, 20],               # dBm
            'maxTxPower': [20, 46],              # dBm
            'basePower': [100, 1000],            # watts
            'idlePower': [50, 500],              # watts
            'dropCallThreshold': [1, 10],        # percentage
            'latencyThreshold': [10, 100],       # ms
            'cpuThreshold': [70, 95],            # percentage
            'prbThreshold': [70, 95],            # percentage
            'trafficLambda': [0.1, 10],          # traffic rate
            'peakHourMultiplier': [1, 5]         # multiplier
        }
        
        # Network features normalization bounds (next 14 features)
        self.network_bounds = {
            'totalEnergy': [0, 10000],           # kWh
            'activeCells': [0, 50],              # number of cells
            'avgDropRate': [0, 20],              # percentage
            'avgLatency': [0, 200],              # ms
            'totalTraffic': [0, 5000],           # traffic units
            'connectedUEs': [0, 200],
            'connectionRate': [0, 100],          # number of UEs
            'cpuViolations': [0, 50],            # number of violations
            'prbViolations': [0, 50],            # number of violations
            'maxCpuUsage': [0, 100],             # percentage
            'maxPrbUsage': [0, 100],
            'kpiViolations': [0, 100],            # percentage
            'totalTxPower': [0, 1000],           # total power
            'avgPowerRatio': [0, 1]              # ratio
        }
        
        # Cell features normalization bounds (12 features per cell)
        self.cell_bounds = {
            'cpuUsage': [0, 100],                # percentage
            'prbUsage': [0, 100],                # percentage
            'currentLoad': [0, 1000],            # load units
            'maxCapacity': [0, 1000],            # capacity units
            'numConnectedUEs': [0, 50],          # number of UEs
            'txPower': [0, 46],                  # dBm
            'energyConsumption': [0, 5000],      # watts
            'avgRSRP': [-140, -70],              # dBm
            'avgRSRQ': [-20, 0],                 # dB
            'avgSINR': [-10, 30],                # dB
            'totalTrafficDemand': [0, 500],      # traffic units
            'loadRatio': [0, 1]                  # ratio
        }
    
    def normalize(self, state_vector):
        """Normalize state vector to [0, 1] range"""
        normalized = np.zeros_like(state_vector)
        
        # Normalize simulation features (first 18)
        simulation_keys = list(self.simulation_bounds.keys())
        for i, key in enumerate(simulation_keys):
            if i < len(state_vector):
                min_val, max_val = self.simulation_bounds[key]
                normalized[i] = self._normalize_value(state_vector[i], min_val, max_val)
        
        # Normalize network features (next 14)
        network_keys = list(self.network_bounds.keys())
        for i, key in enumerate(network_keys):
            global_idx = 18 + i
            if global_idx < len(state_vector):
                min_val, max_val = self.network_bounds[key]
                normalized[global_idx] = self._normalize_value(state_vector[global_idx], min_val, max_val)
        
        # Normalize cell features (remaining features in groups of 12)
        cell_keys = list(self.cell_bounds.keys())
        start_idx = 18 + 14  # After simulation and network features
        
        for cell_idx in range(self.n_cells):
            for feat_idx, key in enumerate(cell_keys):
                global_idx = start_idx + cell_idx * 12 + feat_idx
                if global_idx < len(state_vector):
                    min_val, max_val = self.cell_bounds[key]
                    normalized[global_idx] = self._normalize_value(
                        state_vector[global_idx], min_val, max_val)
        
        return normalized
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize single value to [0, 1] range"""
        if max_val == min_val:
            return 0.5  # Default middle value
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def update_stats(self, state_vector):
        """Update running statistics (implement if using running normalization)"""
        # Optional: Update running mean/std statistics
        pass