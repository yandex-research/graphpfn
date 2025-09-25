import dgl
import torch


class SimpleEdgeSampler:
    def __init__(
        self,
        prob,
        directed: bool = True,
        max_sample_size: int = 1_000_000,
    ):
        super().__init__()
        assert directed is not None
        self._prob = prob
        self._directed = directed
        self.max_sample_size = max_sample_size

    def find_reversed_edges(self, graph, eids):
        src, dst = graph.find_edges(eids.to(torch.int32))
        return graph.edge_ids(dst, src)

    def __call__(self, graph: dgl.DGLGraph):
        edges = torch.arange(graph.num_edges(), device=graph.device)

        if not self._directed:
            src, dst = graph.edges()
            edges = edges[src < dst]

        sample_size = int(len(edges) * self._prob)
        sample_size = min(sample_size, self.max_sample_size)
        sample_ids = torch.randperm(len(edges), device=edges.device)[:sample_size]
        sample = edges[sample_ids]

        if not self._directed:
            sample = torch.cat(
                [
                    sample,
                    self.find_reversed_edges(graph, sample),
                ],
                dim=0,
            )

        return sample
