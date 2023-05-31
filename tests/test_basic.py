import FlowSOM


def test_package_has_version():
    FlowSOM.__version__


def test_fcs(FlowSOM_res):
    assert FlowSOM_res.get_cell_data().shape == (19225, 18)


def test_FlowSOM_type(FlowSOM_res):
    assert isinstance(FlowSOM_res, FlowSOM.main.FlowSOM)


def test_plot_stars(FlowSOM_res):
    FlowSOM.pl.plot_stars(
        FlowSOM_res,
        background_values=FlowSOM_res.get_cluster_data().obs["metaclustering"],
        view="grid",
        equal_node_size=True,
    )


def test_plot_marker(FlowSOM_res):
    FlowSOM.pl.plot_marker(
        FlowSOM_res,
        marker=["CD3"],
        background_values=FlowSOM_res.get_cluster_data().obs["metaclustering"],
        view="grid",
        equal_node_size=True,
    )


def test_new_data(fcs):
    fsom = FlowSOM.main.FlowSOM(fcs[0:5000, :], cols_to_use=[8, 11, 13, 14, 15, 16, 17])
    fsom_new = fsom.new_data(fcs[1001:2000])
    assert fsom_new.get_cell_data().shape == (999, 18)


def test_FlowSOM_class(FlowSOM_res):
    cell_data = FlowSOM_res.get_cell_data()
    uns_true_keys_cell = ["n_nodes", "n_metaclusters"]
    obs_true_keys_cell = ["clustering", "distance_to_bmu", "metaclustering"]
    var_true_keys_cell = ["pretty_colnames", "cols_used", "markers", "channels"]

    length_uns_cell = len(uns_true_keys_cell) == len(cell_data.uns.keys())
    length_obs_cell = len(obs_true_keys_cell) == len(cell_data.obs.keys())
    length_var_cell = len(var_true_keys_cell) == len(cell_data.var.keys())

    uns_keys_cell = all([key in uns_true_keys_cell for key in cell_data.uns.keys()])
    obs_keys_cell = all([key in obs_true_keys_cell for key in cell_data.obs.keys()])
    var_keys_cell = all([key in var_true_keys_cell for key in cell_data.var.keys()])

    cluster_data = FlowSOM_res.get_cluster_data()
    uns_true_keys_cluster = ["outliers", "graph", "xdim", "ydim", "metacluster_MFIs"]
    obs_true_keys_cluster = ["percentages", "metaclustering"]
    obsm_true_keys_cluster = ["cv_values", "sd_values", "mad_values", "codes", "grid", "layout"]

    length_uns_cluster = len(uns_true_keys_cluster) == len(cluster_data.uns.keys())
    length_obs_cluster = len(obs_true_keys_cluster) == len(cluster_data.obs.keys())
    length_obsm_cluster = len(obsm_true_keys_cluster) == len(cluster_data.obsm.keys())

    uns_keys_cluster = all([key in uns_true_keys_cluster for key in cluster_data.uns.keys()])
    obs_keys_cluster = all([key in obs_true_keys_cluster for key in cluster_data.obs.keys()])
    obsm_keys_cluster = all([key in obsm_true_keys_cluster for key in cluster_data.obsm.keys()])
    assert all(
        [
            length_uns_cell,
            length_obs_cell,
            length_var_cell,
            uns_keys_cell,
            obs_keys_cell,
            var_keys_cell,
            length_uns_cluster,
            length_obs_cluster,
            length_obsm_cluster,
            uns_keys_cluster,
            obs_keys_cluster,
            obsm_keys_cluster,
        ]
    )
