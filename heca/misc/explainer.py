from torch_geometric.explain import Explainer


class HoopgnExplainer(Explainer):
    def __call__(self, x_dict, edge_index_dict, edge_attr_dict, *args, **kwargs):
        self.model.set_edge_attr_dict(edge_attr_dict)
        return super().__call__(
            x_dict, edge_index_dict, allow_unused=True, *args, **kwargs
        )
