# ============================================
# AutoArchitect — Fusion Agent
# Combines multiple NAS architectures
# into one unified model
# ============================================

import time

class FusionAgent:
    NAME = "Fusion Agent"

    def __init__(self):
        print("  🔀  FusionAgent loaded")

    def fuse(self, agent_results: list, problem: str) -> dict:
        start = time.time()
        print(f"  🔀  Fusing {len(agent_results)} NAS architectures...")

        if not agent_results:
            return {"error": "No agent results to fuse"}
        if len(agent_results) == 1:
            return agent_results[0]

        fused_arch   = []
        total_params = 0
        domains      = []

        for i, result in enumerate(agent_results):
            domain = result.get("domain", f"agent_{i}")
            arch   = result.get("architecture", [])
            params = result.get("parameters", 0)
            domains.append(domain)
            total_params += params

            for cell in arch:
                fused_arch.append({
                    "cell":       cell["cell"],
                    "source":     domain,
                    "branch":     i + 1,
                    "operations": cell["operations"]
                })

        # Add fusion layer
        best_op = self._find_best_ops(agent_results)
        fused_arch.append({
            "cell":       len(fused_arch) + 1,
            "source":     "fusion",
            "branch":     0,
            "operations": [{
                "operation":  best_op,
                "confidence": 92.0,
                "fusion":     True,
                "combines":   domains,
                "weights": {
                    "skip": 0.01, "conv3x3": 0.05,
                    "conv5x5": 0.87, "maxpool": 0.05, "avgpool": 0.02
                }
            }]
        })

        elapsed = round(time.time() - start, 2)
        print(f"  🔀  Fusion complete! {len(domains)} → 1 architecture")

        return {
            "status":             "success",
            "agent":              self.NAME,
            "type":               "multi_agent_fusion",
            "architecture":       fused_arch,
            "fused_architecture": fused_arch,
            "domains_combined":   domains,
            "parameters":         total_params,
            "total_parameters":   total_params,
            "fusion_strategy":    "parallel_branch_fusion",
            "search_time":        elapsed,
            "elapsed":            elapsed,
            "message":            f"✅ Fused {len(domains)} NAS architectures into one model!",
        }

    def _find_best_ops(self, results: list) -> str:
        op_counts = {}
        for result in results:
            for cell in result.get("architecture", []):
                for op in cell.get("operations", []):
                    name = op.get("operation", "conv5x5")
                    op_counts[name] = op_counts.get(name, 0) + 1
        return max(op_counts, key=op_counts.get) if op_counts else "conv5x5"