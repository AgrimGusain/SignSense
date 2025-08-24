import torch
import pennylane as qml
from torch import nn

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits)}

class DenoiseQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, n_qubits, kernel_size=3, padding=1)
        self.q_weights = nn.Parameter(torch.randn(3, n_qubits))  # Learnable QNN weights

        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 32 * 32),  # Output size matches clean image
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # (B, 4, 1, 1)
        x = x.view(x.size(0), n_qubits)  # (B, 4)

        # QNode batch loop
        results = []
        for i in range(x.shape[0]):
            q_out = quantum_circuit(x[i], self.q_weights)
            q_out = torch.tensor(q_out, dtype=torch.float32, device=x.device)
            results.append(q_out)
        x = torch.stack(results)  # (B, 4)

        x = self.decoder(x)  # (B, 3*32*32)
        x = x.view(-1, 3, 32, 32)  # Reshape to image
        return x
