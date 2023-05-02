import FlowSOM


def test_package_has_version():
    FlowSOM.__version__


def test_fcs(FlowSOM_res):
    assert FlowSOM_res.adata.shape == (19225, 18)


def test_FlowSOM_class_adata(FlowSOM_res):
    assert isinstance(FlowSOM_res, FlowSOM.main.FlowSOM)


def test_plot_stars(FlowSOM_res):
    FlowSOM.pl.plot_stars(FlowSOM_res)


def test_new_data(fcs):
    fsom = FlowSOM.main.FlowSOM(fcs[0:5000, :], cols_to_use=[8, 11, 13, 14, 15, 16, 17])
    fsom_new = fsom.new_data(fcs[1001:2000])
    assert fsom_new.adata.shape == (999, 18)


def test_FlowSOM_class(FlowSOM_res):
    fsom_adata = FlowSOM_res.adata
    uns_true_keys_cell = [
        "pretty_colnames",
        "cols_used",
        "xdim",
        "ydim",
        "n_nodes",
        "cluster_adata",
        "n_metaclusters",
        "metacluster_MFIs",
    ]
    obs_true_keys_cell = ["clustering", "mapping", "metaclustering"]

    length_uns_cell = len(uns_true_keys_cell) == fsom_adata.uns.keys()
    length_obs_cell = len(obs_true_keys_cell) == fsom_adata.obs.keys()

    uns_keys_cell = all([key in uns_true_keys_cell for key in fsom_adata.uns.keys()])
    obs_keys_cell = all([key in obs_true_keys_cell for key in fsom_adata.obs.keys()])

    fsom_cluster_adata = FlowSOM_res.get_cluster_adata()
    uns_true_keys_cluster = ["outliers", "graph"]
    obs_true_keys_cluster = ["percentages", "metaclustering"]
    obsm_true_keys_cluster = ["cv_values", "sd_values", "mad_values", "codes", "grid", "layout"]

    length_uns_cluster = len(uns_true_keys_cluster) == fsom_cluster_adata.uns.keys()
    length_obs_cluster = len(obs_true_keys_cluster) == fsom_cluster_adata.obs.keys()
    length_obsm_cluster = len(obsm_true_keys_cluster) == fsom_cluster_adata.obsm.keys()

    uns_keys_cluster = all([key in uns_true_keys_cluster for key in fsom_cluster_adata.uns.keys()])
    obs_keys_cluster = all([key in obs_true_keys_cluster for key in fsom_cluster_adata.obs.keys()])
    obsm_keys_cluster = all([key in obsm_true_keys_cluster for key in fsom_cluster_adata.obsm.keys()])
    assert all(
        [
            length_uns_cell,
            length_obs_cell,
            uns_keys_cell,
            obs_keys_cell,
            length_uns_cluster,
            length_obs_cluster,
            length_obsm_cluster,
            uns_keys_cluster,
            obs_keys_cluster,
            obsm_keys_cluster,
        ]
    )
